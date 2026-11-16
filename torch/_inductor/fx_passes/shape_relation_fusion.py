import operator

import sympy

import torch
from torch._guards import CompileContext, TracingContext
from torch.fx.experimental.symbolic_shapes import (
    guard_or_false,
    statically_known_true,
    sym_and,
    sym_eq,
)
from torch.utils._ordered_set import OrderedSet

from ..virtualized import V


def canonical_shape_key(shape, allocators):
    """Hashable dimensions using existing facts, without adding guards."""
    from ..sizevars import SizeVarAllocator

    result = []
    for dim in shape:
        if isinstance(dim, torch.SymInt):
            env = dim.node.shape_env
            if env is None:
                return None
            if env not in allocators:
                allocators[env] = SizeVarAllocator(env)
            expr = allocators[env].simplify(dim.node.expr)
            # Symbol names may repeat in distinct ShapeEnvs. Constants do not
            # need an environment identity; symbolic dimensions do.
            result.append(expr if expr.is_number else (env, expr))
        else:
            result.append(dim)
    return tuple(result)


def guard_shape_compatibility(shapes, extra_condition, max_conditions):
    """Guard one conjunction so all incompatible shapes share its negation.

    The caller owns candidate legality and profitability. Never speculate on
    unbacked sizes or when there is no Dynamo dispatcher for the fallback.
    """
    context = TracingContext.try_get()
    if (
        V.aot_compilation
        or CompileContext.try_get() is None
        or context is None
        or context.fake_mode is None
        or context.fake_mode.shape_env is None
        or context.fake_mode.shape_env.frozen
    ):
        return False
    env = context.fake_mode.shape_env
    if not shapes or any(len(shape) != len(shapes[0]) for shape in shapes):
        return False
    for shape in shapes:
        for dim in shape:
            if isinstance(dim, torch.SymInt) and (
                dim.node.shape_env is not env
                or not isinstance(dim.node.expr, sympy.Symbol)
                or dim.node.expr not in env.backed_var_to_val
            ):
                return False
            if not statically_known_true(dim > 1):
                return False
    predicates = []
    seen = OrderedSet()
    for shape in shapes[1:]:
        for left, right in zip(shapes[0], shape):
            predicate = left == right
            if statically_known_true(predicate):
                continue
            key = (
                predicate.node.expr
                if isinstance(predicate, torch.SymBool)
                else predicate
            )
            if key not in seen:
                seen.add(key)
                predicates.append(predicate)
    if not statically_known_true(extra_condition):
        predicates.append(extra_condition)
    if len(predicates) > max_conditions:
        return False
    return guard_or_false(sym_and(True, *predicates))


def _reduction_shape(node):
    """Find a kept prefix and a reduced suffix from operator metadata."""
    if (
        node.op != "call_function"
        or not isinstance(node.target, torch._ops.OpOverload)
        or torch.Tag.reduction not in node.target.tags
        or node.target._schema.is_mutable
        or not node.args
        or not isinstance(node.args[0], torch.fx.Node)
    ):
        return None
    source = node.args[0].meta.get("val")
    if (
        not isinstance(source, torch.Tensor)
        or source.layout != torch.strided
        or source.device.type not in ("cuda", "mps")
        or not source.dtype.is_floating_point
        or source.ndim < 2
    ):
        return None
    value = node.meta.get("val")
    roots = (
        [node]
        if isinstance(value, torch.Tensor)
        else [
            user
            for user in node.users
            if user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) == 2
            and isinstance(user.args[1], int)
        ]
    )
    for root in roots:
        output = root.meta.get("val")
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != source.ndim
            or output.dtype != source.dtype
            or output.device != source.device
        ):
            continue
        axes = [
            i
            for i, (left, right) in enumerate(zip(source.shape, output.shape))
            if not statically_known_true(left == right)
        ]
        if (
            not axes
            or axes[0] == 0
            or axes != list(range(axes[0], source.ndim))
            or not all(statically_known_true(output.shape[i] == 1) for i in axes)
        ):
            continue
        return source, axes[0], roots
    return None


def guard_reduction_shape_equality(graph: torch.fx.Graph) -> None:
    """Specialize one reduction epilogue on a joint iteration-shape guard.

    Operator tags and tensor metadata describe the reduction, rather than an
    operator-name list or a particular schema's dim/keepdim argument positions.
    The shared guard helper preserves the unequal-shape fallback.
    """
    for reduction in graph.nodes:
        candidate = _reduction_shape(reduction)
        if candidate is None:
            continue
        source, prefix_rank, roots = candidate
        pending = list(roots)
        seen = OrderedSet()
        while pending and len(seen) < 8:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            value = current.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != source.ndim
                or value.device != source.device
                or value.dtype != source.dtype
                or not statically_known_true(
                    sym_eq(value.shape[:prefix_rank], source.shape[:prefix_rank])
                )
            ):
                continue
            if all(statically_known_true(d == 1) for d in value.shape[prefix_rank:]):
                pending.extend(
                    user
                    for user in current.users
                    if user.op == "call_function"
                    and isinstance(user.target, torch._ops.OpOverload)
                    and torch.Tag.pointwise in user.target.tags
                    and not user.target._schema.is_mutable
                )
                continue
            left, right = source.shape[prefix_rank:], value.shape[prefix_rank:]
            unknown = [
                (a, b) for a, b in zip(left, right) if not statically_known_true(a == b)
            ]
            if not unknown or any(
                not isinstance(a, torch.SymInt)
                or not isinstance(b, torch.SymInt)
                or statically_known_true(a != b)
                for a, b in unknown
            ):
                continue
            # One conjunction covers different mismatch axes with one fallback.
            guard_shape_compatibility([left, right], True, max_conditions=4)
            return
