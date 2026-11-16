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


def guard_reduction_width_equality(graph: torch.fx.Graph) -> None:
    """Specialize one row-reduction epilogue on a backed width equality.

    Both equality and inequality are guarded: an unequal-first call must not
    permanently hide the fused specialization. This only changes Dynamo's
    specialization domain, never the accepted input domain. AOT exports have
    no Dynamo guard dispatcher and must not acquire these assumptions.
    """
    context = TracingContext.try_get()
    if (
        V.aot_compilation
        or CompileContext.try_get() is None
        or context is None
        or context.fake_mode is None
    ):
        return

    for reduction in graph.nodes:
        if reduction.op != "call_function" or reduction.target not in (
            torch.ops.aten.mean.dim,
            torch.ops.aten.sum.dim_IntList,
        ):
            continue
        arg = reduction.args[0]
        if not isinstance(arg, torch.fx.Node):
            continue
        source = arg.meta.get("val")
        dims = (
            reduction.args[1]
            if len(reduction.args) > 1
            else reduction.kwargs.get("dim")
        )
        keepdim = (
            reduction.args[2]
            if len(reduction.args) > 2
            else reduction.kwargs.get("keepdim", False)
        )
        if (
            not isinstance(source, torch.Tensor)
            or source.device.type not in ("cuda", "mps")
            or not source.dtype.is_floating_point
            or source.ndim < 2
            or dims not in ([-1], [source.ndim - 1])
            or keepdim is not True
        ):
            continue

        # Follow only pointwise operations that still have the reduced shape.
        # Stop at the first full-width consumer; bound exploration and guards.
        pending = list(reduction.users)
        seen = OrderedSet()
        while pending and len(seen) < 8:
            user = pending.pop()
            if user in seen:
                continue
            seen.add(user)
            if (
                user.op != "call_function"
                or not isinstance(user.target, torch._ops.OpOverload)
                or torch.Tag.pointwise not in user.target.tags
            ):
                continue
            value = user.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != source.ndim
                or value.device != source.device
                or value.dtype != source.dtype
                or not statically_known_true(
                    sym_eq(value.shape[:-1], source.shape[:-1])
                )
            ):
                continue
            if statically_known_true(value.shape[-1] == 1):
                pending.extend(user.users)
                continue
            left, right = source.shape[-1], value.shape[-1]
            if not isinstance(left, torch.SymInt) or not isinstance(
                right, torch.SymInt
            ):
                continue
            env = left.node.shape_env
            if (
                env is None
                or env is not context.fake_mode.shape_env
                or env is not right.node.shape_env
                or env.frozen
                or not isinstance(left.node.expr, sympy.Symbol)
                or not isinstance(right.node.expr, sympy.Symbol)
                or left.node.expr not in env.backed_var_to_val
                or right.node.expr not in env.backed_var_to_val
                or not statically_known_true(left > 1)
                or not statically_known_true(right > 1)
                or statically_known_true(left == right)
                or statically_known_true(left != right)
            ):
                continue
            # On equality ShapeEnv unifies the symbols before lowering. On
            # inequality the ordinary unfused graph remains a valid fallback.
            guard_shape_compatibility([(left,), (right,)], True, max_conditions=1)
            return
