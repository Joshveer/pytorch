# Owner(s): ["module: inductor"]
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
import torch.nn.functional as F
from torch._dynamo.source import LocalSource
from torch._dynamo.utils import counters
from torch._guards import compile_context, CompileContext, tracing, TracingContext
from torch._inductor import config
from torch._inductor.codegen.simd import CantSplit, SIMDKernel, SIMDScheduling
from torch._inductor.dependencies import MemoryDep
from torch._inductor.fx_passes.shape_relation_fusion import (
    guard_reduction_width_equality,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.tiling_utils import get_pw_red_splits
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, Max, Mod


class TestShapeRelationSimplify(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("lower", [0, 2, 4])
    @parametrize("divisible", [False, True])
    def test_unit_max_uses_proof(self, lower, divisible):
        env = ShapeEnv()
        h = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(h, lower, 64)
        if divisible:
            env._maybe_guard_rel(sympy.Eq(Mod(h, 4), 0))
        sv = SizeVarAllocator(env)
        q = FloorDiv(h, 4)
        before = (list(env.guards), dict(env.replacements), set(env.divisible))
        result = sv.simplify(Max(1, q))
        proved_positive = lower >= 4 or (lower > 0 and divisible)
        self.assertEqual(result, q if proved_positive else Max(1, q))
        for value in range(lower, 65):
            if not divisible or value % 4 == 0:
                self.assertEqual(result.subs(h, value), Max(1, q).subs(h, value))
        self.assertEqual(
            before, (list(env.guards), dict(env.replacements), set(env.divisible))
        )

    def test_scaled_divisibility_and_range_split(self):
        env = ShapeEnv()
        h = env.create_unbacked_symint().node.expr
        other = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(h, 2, 64)
        sv = SizeVarAllocator(env)
        q = FloorDiv(h, 4)
        with V.set_graph_handler(SimpleNamespace(sizevars=sv)):
            with self.assertRaises(CantSplit):
                SIMDKernel._split_iteration_ranges([8 * h], [[32, q]])
            env._maybe_guard_rel(sympy.Eq(Mod(h, 4), 0))
            before = (list(env.guards), dict(env.replacements), set(env.divisible))
            self.assertTrue(sv.statically_known_multiple_of(2048 * h, 8192))
            self.assertFalse(sv.statically_known_multiple_of(2048 * h, 6144))
            self.assertFalse(sv.statically_known_multiple_of(2048 * other, 8192))
            ranges, getters = SIMDKernel._split_iteration_ranges([8 * h], [[32, q]])
            self.assertEqual(ranges, [[32, q]])
            for value in range(4, 65, 4):
                coordinates = [
                    tuple(f([i, j]) for f in getters[0])
                    for i in range(32)
                    for j in range(value // 4)
                ]
                self.assertEqual(
                    [i * (value // 4) + j for i, j in coordinates],
                    list(range(8 * value)),
                )
            self.assertEqual(
                before, (list(env.guards), dict(env.replacements), set(env.divisible))
            )

    def test_group_equality_and_dependency_normalization(self):
        env = ShapeEnv()
        h = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(h, 2, 64)
        env._maybe_guard_rel(sympy.Eq(Mod(h, 4), 0))
        sv = SizeVarAllocator(env)
        self.assertTrue(sv.statically_known_equals((h, 1), (h, 1)))
        self.assertFalse(sv.statically_known_equals((h, 1), (h, 2)))
        q = FloorDiv(h, 4)
        i, j, k = sympy.symbols("i j k", integer=True, nonnegative=True)
        with V.set_graph_handler(SimpleNamespace(sizevars=sv)):
            self.assertTrue(sv.statically_known_equals(32 * q, 8 * h))
            node = SimpleNamespace(
                is_reduction=lambda: False,
                _body=SimpleNamespace(
                    sizes=([32, q], []), iter_vars=[i, j], reduce_vars=[]
                ),
            )
            self.assertEqual(
                get_pw_red_splits(node, 8 * h, sympy.Integer(16)),
                (([i, j], [32, q]), ([], [])),
            )
            flat = MemoryDep("buf", i * h + j, (i, j), (8, h)).normalize()
            split = MemoryDep(
                "buf", i * h + j * q + k, (i, j, k), (8, 4, q)
            ).normalize()
            shifted = MemoryDep(
                "buf", i * h + j * q + k + 1, (i, j, k), (8, 4, q)
            ).normalize()
            self.assertEqual(flat, split)
            self.assertNotEqual(flat, shifted)

    def test_cache_tracks_new_divisibility_and_bounds(self):
        env = ShapeEnv()
        h = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(h, 2, 64)
        sv = SizeVarAllocator(env)
        q = FloorDiv(h, 4)
        i, j, k = sympy.symbols("i j k", integer=True, nonnegative=True)
        self.assertEqual(sv.simplify_with_ranges(Max(1, q), {}), Max(1, q))
        before_sizes = sv._simplify_loops([i, j, k], [8, 4, q], [i * h + j * q + k])[0]
        env._maybe_guard_rel(sympy.Eq(Mod(h, 4), 0))
        self.assertEqual(sv.simplify_with_ranges(Max(1, q), {}), q)
        self.assertNotEqual(before_sizes, [8 * h])
        self.assertEqual(
            sv._simplify_loops([i, j, k], [8, 4, q], [i * h + j * q + k])[0], [8 * h]
        )
        other = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(other, 2, 64)
        expr = Max(1, FloorDiv(other, 4))
        self.assertEqual(sv.stride_vars(i * expr, [i]), [expr])
        env.constrain_symbol_range(other, 4, 64)
        self.assertEqual(sv.stride_vars(i * expr, [i]), [FloorDiv(other, 4)])

    @parametrize("divisor", [2, 4, 8, 16])
    def test_proven_divisibility_cancels_product(self, divisor):
        env = ShapeEnv()
        c = env.create_unbacked_symint().node.expr
        t = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(c, 0, 1024)
        env.constrain_symbol_range(t, 0, 1024)
        sv = SizeVarAllocator(env)
        expr = 2 * divisor * t * FloorDiv(c, divisor)
        self.assertEqual(sv.simplify(expr), expr)
        env._maybe_guard_rel(sympy.Eq(Mod(c, divisor), 0))
        before = (list(env.guards), dict(env.replacements), set(env.divisible))
        self.assertEqual(sv.simplify(expr), 2 * t * c)
        # Exercise the actual grouping entry point, not just the helper.
        with V.set_graph_handler(SimpleNamespace(sizevars=sv)):
            self.assertEqual(
                SIMDScheduling.group_fn(
                    None, [[2, t, divisor, FloorDiv(c, divisor)], []]
                ),
                (2 * t * c, 1),
            )
        self.assertEqual(
            before, (list(env.guards), dict(env.replacements), set(env.divisible))
        )

    def test_missing_or_unrelated_fact_does_not_cancel(self):
        env = ShapeEnv()
        c = env.create_unbacked_symint().node.expr
        sv = SizeVarAllocator(env)
        expr = 8 * FloorDiv(c, 8)
        self.assertEqual(sv.simplify(expr), expr)
        env._maybe_guard_rel(sympy.Eq(Mod(c, 3), 0))
        self.assertEqual(sv.simplify(expr), expr)
        # 9 satisfies the known fact but not the proposed cancellation.
        self.assertEqual(sv.simplify(expr).subs(c, 9), 8)

    def test_do_not_cancel_partial_coefficient_or_unknown_symbol(self):
        env = ShapeEnv()
        c = env.create_unbacked_symint().node.expr
        other = env.create_unbacked_symint().node.expr
        sv = SizeVarAllocator(env)
        env._maybe_guard_rel(sympy.Eq(Mod(c, 8), 0))
        self.assertEqual(sv.simplify(4 * FloorDiv(c, 8)), 4 * FloorDiv(c, 8))
        self.assertEqual(sv.simplify(8 * FloorDiv(other, 8)), 8 * FloorDiv(other, 8))
        self.assertFalse(sv.statically_known_equals(c, other))

    def test_nested_products_and_nonmultiples(self):
        env = ShapeEnv()
        c = env.create_unbacked_symint().node.expr
        sv = SizeVarAllocator(env)
        env._maybe_guard_rel(sympy.Eq(Mod(c, 8), 0))
        expr = (16 * FloorDiv(c, 8) + 3) ** 2
        result = sv.simplify(expr)
        for value in range(0, 1025, 8):
            self.assertEqual(result.subs(c, value), expr.subs(c, value))


class TestSplitHeadFusion(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("kind", ["softmax", "rmsnorm", "layernorm"])
    def test_dynamic_split_forward_backward(self, device, kind):
        def fn(x):
            b, t, c = x.shape
            if kind == "softmax":
                y = x.softmax(-1)
            elif kind == "rmsnorm":
                y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            else:
                y = torch.nn.functional.layer_norm(x, (c,))
            return y.view(b, t, 8, c // 8) * 2

        torch._dynamo.reset()
        with config.patch(force_disable_caches=True):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            for width in (128, 256, 24, 8):
                x = torch.randn(2, 17, width, device=device, requires_grad=True)
                torch._dynamo.mark_dynamic(x, 1)
                # The singleton head width may specialize.
                if width != 8:
                    torch._dynamo.mark_dynamic(x, 2)
                actual = compiled(x)
                expected = fn(x)
                self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
                weight = torch.randn_like(expected)
                actual_grad = torch.autograd.grad(actual, x, weight)[0]
                expected_grad = torch.autograd.grad(expected, x, weight)[0]
                self.assertEqual(actual_grad, expected_grad, atol=2e-4, rtol=2e-4)

    def test_one_dynamic_softmax_split_kernel(self, device):
        if torch.device(device).type == "cpu":
            self.skipTest("Kernel-count regression is for SIMD GPU backends")

        def fn(x):
            b, t, c = x.shape
            return x.softmax(-1).view(b, t, 8, c // 8) * 2

        torch._dynamo.reset()
        x = torch.randn(2, 512, 512, device=device)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(x, 2)
        with config.patch(force_disable_caches=True):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, fn(x), atol=1e-4, rtol=1e-4)
        backend = "metal" if torch.device(device).type == "mps" else "triton"
        self.assertEqual(
            sum(code.count(f"= async_compile.{backend}(") for code in codes), 1
        )


class TestGuardedShapeFusion(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_compiler_cache_preserves_equality_guards(self, device):
        def fn(x, y):
            return y * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

        hits_before = counters["inductor"]["fxgraph_cache_hit"]
        with (
            fresh_cache(),
            config.patch(force_disable_caches=False, fx_graph_cache=True),
        ):
            # Reset only Dynamo between rounds, leaving the Inductor disk cache.
            for _ in range(2):
                torch._dynamo.reset()
                compiled = torch.compile(fn, backend="inductor", fullgraph=True)
                for left, right in [(128, 128), (96, 160), (256, 256), (72, 144)]:
                    x = torch.randn(17, left, device=device)
                    y = torch.randn(17, right, device=device)
                    for tensor in (x, y):
                        torch._dynamo.mark_dynamic(tensor, 0)
                        torch._dynamo.mark_dynamic(tensor, 1)
                    self.assertEqual(compiled(x, y), fn(x, y), atol=1e-4, rtol=1e-4)
        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], hits_before)

    @parametrize("training", [False, True])
    @parametrize("strided", [False, True])
    @parametrize("unequal_first", [False, True])
    def test_equal_and_unequal_widths(self, device, training, strided, unequal_first):
        def fn(x, y):
            return y * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        torch._dynamo.reset()
        widths = [(128, 128), (96, 160), (256, 256), (72, 144), (128, 128)]
        if unequal_first:
            widths[0], widths[1] = widths[1], widths[0]
        gpu = torch.device(device).type != "cpu"
        with config.patch(force_disable_caches=True, guarded_shape_fusion=True):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for left, right in widths:
                inputs = []
                for width in (left, right):
                    x = torch.randn(17, width * (2 if strided else 1), device=device)
                    if strided:
                        x = x[:, ::2]
                    x.requires_grad_(training)
                    torch._dynamo.mark_dynamic(x, 0)
                    torch._dynamo.mark_dynamic(x, 1)
                    inputs.append(x)
                codes = []
                # Preserve Dynamo's cache across calls; run_and_get_code resets it.
                with patch.object(GraphLowering, "save_output_code", codes.append):
                    actual = compiled(*inputs)
                    expected = fn(*inputs)
                    self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
                    if training:
                        weight = torch.randn_like(actual)
                        actual_grads = torch.autograd.grad(actual, inputs, weight)
                        expected_grads = torch.autograd.grad(expected, inputs, weight)
                        self.assertEqual(
                            actual_grads, expected_grads, atol=2e-4, rtol=2e-4
                        )
                if gpu and codes:
                    backend_name = (
                        "metal" if torch.device(device).type == "mps" else "triton"
                    )
                    self.assertEqual(
                        codes[0].count(f"= async_compile.{backend_name}("),
                        1 if left == right else 2,
                    )
            self.assertEqual(len(graphs), 2 if gpu else 1)

    @parametrize(
        "rows,left,right", [(0, 8, 8), (1, 8, 16), (3, 1, 1), (3, 1, 8), (3, 8, 1)]
    )
    def test_degenerate_dimensions(self, device, rows, left, right):
        def fn(x, y):
            return y * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

        torch._dynamo.reset()
        x = torch.randn(rows, left, device=device, requires_grad=True)
        y = torch.randn(rows, right, device=device, requires_grad=True)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(x, y)
        expected = fn(x, y)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        weight = torch.randn_like(actual)
        self.assertEqual(
            torch.autograd.grad(actual, (x, y), weight),
            torch.autograd.grad(expected, (x, y), weight),
            atol=2e-4,
            rtol=2e-4,
        )

    def test_disabled(self, device):
        def fn(x, y):
            return y * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

        torch._dynamo.reset()
        x = torch.randn(17, 128, device=device)
        y = torch.randn_like(x)
        for tensor in (x, y):
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)
        with config.patch(guarded_shape_fusion=False, force_disable_caches=True):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
        self.assertEqual(actual, fn(x, y), atol=1e-4, rtol=1e-4)
        if torch.device(device).type != "cpu":
            backend_name = "metal" if torch.device(device).type == "mps" else "triton"
            self.assertEqual(
                sum(code.count(f"= async_compile.{backend_name}(") for code in codes), 2
            )


class TestEqualityGuardPolicy(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "scenario", ["equal", "unequal", "unbacked", "aot", "frozen", "no_dispatcher"]
    )
    def test_guard_policy(self, scenario):
        env = ShapeEnv()
        widths = []
        for name, value in [
            ("left", 128),
            ("right", 96 if scenario == "unequal" else 128),
        ]:
            if scenario == "unbacked":
                width = env.create_unbacked_symint()
                env.constrain_symbol_range(width.node.expr, 2, 1024)
            else:
                source = LocalSource(name)
                expr = env.create_symbol(value, source, dynamic_dim=DimDynamic.DYNAMIC)
                width = env.create_symintnode(expr, hint=value)
            widths.append(width)
        fake_mode = FakeTensorMode(shape_env=env)
        with fake_mode:
            x_value = torch.empty(17, widths[0], device="mps")
            y_value = torch.empty(17, widths[1], device="mps")
            reduced_value = x_value.mean(-1, keepdim=True)
            result_value = y_value * reduced_value
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = x_value
        y = graph.placeholder("y")
        y.meta["val"] = y_value
        reduction = graph.call_function(torch.ops.aten.mean.dim, (x, [-1], True))
        reduction.meta["val"] = reduced_value
        result = graph.call_function(torch.ops.aten.mul.Tensor, (y, reduction))
        result.meta["val"] = result_value
        graph.output(result)
        if scenario == "frozen":
            env.freeze()
        before = len(env.guards)
        with (
            V.set_aot_compilation(scenario == "aot"),
            compile_context(
                None if scenario == "no_dispatcher" else CompileContext(None)
            ),
            tracing(TracingContext(fake_mode)),
        ):
            guard_reduction_width_equality(graph)
        self.assertEqual(
            len(env.guards) - before, int(scenario in ("equal", "unequal"))
        )
        self.assertEqual(
            widths[0].node.expr == widths[1].node.expr, scenario == "equal"
        )


# Workloads from CyrusD123/inductor-dynamic-fusion, commit 275f764.
def xsa_unsqueeze(output, xv):
    head_dim = xv.shape[-1]
    n_kv = xv.shape[2]
    n_heads = output.shape[2]
    torch._check(output.shape[3] == head_dim)
    torch._check(n_heads % n_kv == 0)
    n_rep = n_heads // n_kv

    vn = F.normalize(xv, dim=-1, eps=1e-6).unsqueeze(3)
    y = output.view(*output.shape[:2], n_kv, n_rep, head_dim)
    ps = (y * vn).sum(dim=-1, keepdim=True)
    cos_sim = ps.squeeze(-1).abs()
    proj_frac = cos_sim / y.norm(dim=-1).clamp(min=1e-6)
    result = (y - ps * vn).view_as(output)
    return result, cos_sim, proj_frac


def xsa_expand(output, xv):
    B, T, Hq, D = output.shape
    Hkv = xv.shape[2]
    torch._check(xv.shape[3] == D)
    torch._check(Hq % Hkv == 0)
    gs = Hq // Hkv

    v_exp = xv[:, :, :, None, :].expand(B, T, Hkv, gs, D).reshape(B, T, Hq, D)
    vn = F.normalize(v_exp, p=2, dim=-1, eps=1e-6)
    coeff = (output * vn).sum(dim=-1, keepdim=True)
    projected = output - coeff * vn
    cg = coeff.squeeze(-1).view(B, T, Hkv, gs)
    cos_sim = cg.abs()
    yg = output.view(B, T, Hkv, gs, D)
    proj_frac = cos_sim / yg.norm(dim=-1).clamp(min=1e-6)
    return projected, cos_sim, proj_frac


class TestXSAShapeFusion(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("variant", ["unsqueeze", "expand"])
    @parametrize("inference", [False, True])
    @dtypes(torch.float32, torch.bfloat16)
    def test_original_reproduction(self, device, dtype, variant, inference):
        fn = xsa_unsqueeze if variant == "unsqueeze" else xsa_expand
        static_values = None
        static_counts = None
        dynamic_values = None
        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-4
        configurations = [(False, False), (True, False)]
        if dtype == torch.bfloat16 and not inference:
            configurations.append((True, True))

        def baseline_simplify(self, expr):
            return sympy.expand(expr).xreplace(self.replacements)

        for dynamic, baseline in configurations:
            torch._dynamo.reset()
            torch.manual_seed(1234)
            x = torch.randn(
                1, 2048, 32, 64, device=device, dtype=dtype, requires_grad=not inference
            )
            v = torch.randn(
                1, 2048, 4, 64, device=device, dtype=dtype, requires_grad=not inference
            )
            weights = (
                torch.randn_like(x),
                torch.randn(1, 2048, 4, 8, device=device, dtype=dtype),
                torch.randn(1, 2048, 4, 8, device=device, dtype=dtype),
            )
            if dynamic:
                for tensor, dim in [
                    (x, 2),
                    (x, 3),
                    (v, 3),
                    (weights[0], 2),
                    (weights[0], 3),
                    (weights[1], 3),
                    (weights[2], 3),
                ]:
                    torch._dynamo.mark_dynamic(tensor, dim)

            def fwd_bwd():
                values = fn(x, v)
                if not inference:
                    torch.autograd.backward(list(values), list(weights))
                return values

            with (
                config.patch(
                    force_disable_caches=True, guarded_shape_fusion=not baseline
                ),
                patch.object(SizeVarAllocator, "simplify", baseline_simplify)
                if baseline
                else nullcontext(),
            ):
                actual, codes = run_and_get_code(
                    torch.compile(fwd_bwd),
                )
            expected = fn(x, v)
            self.assertEqual(actual, expected, atol=tolerance, rtol=tolerance)
            values = tuple(t.detach().clone() for t in actual)
            if not inference:
                values += (x.grad.detach().clone(), v.grad.detach().clone())
                # FP32 checks eager VJPs. BF16 dynamic gradients already differ
                # from both eager and compiled-static, so compare them against
                # the unoptimized dynamic compiler rather than masking a change.
                if dtype == torch.float32:
                    grads = torch.autograd.grad(expected, (x, v), weights)
                    self.assertEqual(
                        (x.grad, v.grad), grads, atol=tolerance, rtol=tolerance
                    )
            backend = "metal" if torch.device(device).type == "mps" else "triton"
            counts = [code.count(f"= async_compile.{backend}(") for code in codes]
            if baseline:
                self.assertEqual(values, dynamic_values, atol=tolerance, rtol=tolerance)
            elif dynamic:
                dynamic_values = values
                self.assertEqual(counts, static_counts)
                if dtype == torch.bfloat16 and not inference:
                    self.assertEqual(
                        values[:3], static_values[:3], atol=tolerance, rtol=tolerance
                    )
                else:
                    self.assertEqual(
                        values, static_values, atol=tolerance, rtol=tolerance
                    )
            else:
                static_values, static_counts = values, counts
                if torch.device(device).type == "mps":
                    self.assertEqual(
                        counts,
                        ([2] if inference else [2, 4])
                        if variant == "unsqueeze"
                        else ([1] if inference else [1, 2]),
                    )

    @parametrize("variant", ["unsqueeze", "expand"])
    @parametrize("kv", [2, 4, 8])
    @parametrize("strided", [False, True])
    def test_shape_sweep_gradients(self, device, variant, kv, strided):
        fn = xsa_unsqueeze if variant == "unsqueeze" else xsa_expand
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True)
        for heads, width in [(2 * kv, 16), (3 * kv, 24), (8 * kv, 32), (kv, 8)]:
            inputs = []
            for h in (heads, kv):
                t = torch.randn(1, 19, h, width * (2 if strided else 1), device=device)
                if strided:
                    t = t[..., ::2]
                inputs.append(t.requires_grad_())
                torch._dynamo.mark_dynamic(t, 3)
            if heads != kv:
                torch._dynamo.mark_dynamic(inputs[0], 2)
            actual = compiled(*inputs)
            expected = fn(*inputs)
            self.assertEqual(actual, expected, atol=2e-4, rtol=2e-4)
            weights = tuple(torch.randn_like(v) for v in actual)
            self.assertEqual(
                torch.autograd.grad(actual, inputs, weights),
                torch.autograd.grad(expected, inputs, weights),
                atol=2e-4,
                rtol=2e-4,
            )


instantiate_parametrized_tests(TestShapeRelationSimplify)
instantiate_parametrized_tests(TestEqualityGuardPolicy)
instantiate_device_type_tests(
    TestSplitHeadFusion, globals(), except_for=("xpu", "privateuse1"), allow_mps=True
)
instantiate_device_type_tests(
    TestGuardedShapeFusion, globals(), except_for=("xpu", "privateuse1"), allow_mps=True
)

instantiate_device_type_tests(
    TestXSAShapeFusion,
    globals(),
    except_for=("cpu", "xpu", "privateuse1"),
    allow_mps=True,
)

if __name__ == "__main__":
    run_tests()
