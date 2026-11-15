# Owner(s): ["module: inductor"]
from types import SimpleNamespace
from unittest.mock import patch

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._dynamo.utils import counters
from torch._guards import compile_context, CompileContext, tracing, TracingContext
from torch._inductor import config
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.fx_passes.shape_relation_fusion import (
    guard_reduction_width_equality,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, Mod


class TestShapeRelationSimplify(TestCase):
    hw_classification = HardwareClassification.GENERIC

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


instantiate_parametrized_tests(TestShapeRelationSimplify)
instantiate_parametrized_tests(TestEqualityGuardPolicy)
instantiate_device_type_tests(
    TestSplitHeadFusion, globals(), except_for=("xpu", "privateuse1"), allow_mps=True
)
instantiate_device_type_tests(
    TestGuardedShapeFusion, globals(), except_for=("xpu", "privateuse1"), allow_mps=True
)

if __name__ == "__main__":
    run_tests()
