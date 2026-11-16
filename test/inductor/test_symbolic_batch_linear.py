# Owner(s): ["module: inductor"]
from unittest.mock import patch

import sympy

import torch
import torch.nn.functional as F
from torch._dynamo.source import LocalSource
from torch._dynamo.utils import counters
from torch._guards import compile_context, CompileContext, tracing, TracingContext
from torch._inductor import config
from torch._inductor.fx_passes.group_batch_fusion import PostGradBatchLinearFusion
from torch._inductor.fx_passes.shape_relation_fusion import (
    canonical_shape_key,
    guard_shape_compatibility,
)
from torch._inductor.graph import GraphLowering
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
from torch.utils._sympy.functions import FloorDiv, Mod


def parallel_linear(*args):
    return tuple(F.linear(args[2 * i], args[2 * i + 1]) for i in range(len(args) // 2))


def guarded_options(**kwargs):
    return {"batch_linear_post_grad": {"guarded_batching": True, **kwargs}}


class TestBatchShapeRelations(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_device_keys_do_not_mix(self):
        graph = torch.fx.Graph()
        rule = PostGradBatchLinearFusion()
        keys = []
        with FakeTensorMode():
            for device in ("cpu", "mps"):
                left = graph.placeholder(f"left_{device}")
                right = graph.placeholder(f"right_{device}")
                left.meta["val"] = torch.empty(64, 32, device=device)
                right.meta["val"] = torch.empty(32, 48, device=device)
                node = graph.call_function(torch.ops.aten.mm.default, (left, right))
                node.meta["val"] = torch.empty(64, 48, device=device)
                keys.append(rule.match(node))
        self.assertNotEqual(keys[0], keys[1])

    def test_canonical_keys_use_facts_and_environment(self):
        env = ShapeEnv()
        h = env.create_unbacked_symint()
        env.constrain_symbol_range(h.node.expr, 2, 128)
        env._maybe_guard_rel(sympy.Eq(Mod(h.node.expr, 4), 0))
        allocators = {}
        expr = env.create_symintnode(8 * FloorDiv(h.node.expr, 4), hint=None)
        before = (list(env.guards), dict(env.replacements), set(env.divisible))
        self.assertEqual(
            canonical_shape_key((expr, 16), allocators),
            canonical_shape_key((2 * h, 16), allocators),
        )
        other = ShapeEnv().create_unbacked_symint()
        self.assertNotEqual(
            canonical_shape_key((h, 16), allocators),
            canonical_shape_key((other, 16), allocators),
        )
        self.assertEqual(
            before, (list(env.guards), dict(env.replacements), set(env.divisible))
        )

    @parametrize(
        "scenario",
        [
            "equal",
            "unequal",
            "oversize",
            "budget",
            "unbacked",
            "frozen",
            "aot",
            "no_dispatcher",
        ],
    )
    def test_joint_guard_and_refusal(self, scenario):
        env = ShapeEnv()
        dims = []
        for name, hint in [
            ("a", 64),
            ("b", 80 if scenario == "unequal" else 64),
            ("c", 32),
            ("d", 32),
        ]:
            if scenario == "unbacked":
                value = env.create_unbacked_symint()
                env.constrain_symbol_range(value.node.expr, 2, 128)
            else:
                expr = env.create_symbol(
                    hint, LocalSource(name), dynamic_dim=DimDynamic.DYNAMIC
                )
                value = env.create_symintnode(expr, hint=hint)
            dims.append(value)
        a, b, c, d = dims
        if scenario == "frozen":
            env.freeze()
        context = TracingContext(FakeTensorMode(shape_env=env))
        before = len(env.guards)
        with (
            tracing(context),
            compile_context(
                None if scenario == "no_dispatcher" else CompileContext(None)
            ),
            V.set_aot_compilation(scenario == "aot"),
        ):
            accepted = guard_shape_compatibility(
                [(a, c), (b, d)],
                a * c <= (16 if scenario == "oversize" else 8192),
                0 if scenario == "budget" else 8,
            )
        self.assertEqual(accepted, scenario == "equal")
        self.assertEqual(
            len(env.guards) - before, int(scenario in ("equal", "unequal", "oversize"))
        )
        if scenario in ("unequal", "oversize"):
            self.assertFalse(env.replacements)


class TestSymbolicBatchLinear(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(torch.float16, torch.bfloat16)
    def test_guarded_low_precision_gradients(self, device, dtype):
        if (
            dtype == torch.bfloat16
            and torch.device(device).type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            self.skipTest("BF16 unsupported")
        torch._dynamo.reset()
        args = tuple(
            t.requires_grad_()
            for _ in range(5)
            for t in (
                torch.randn(64, 32, device=device, dtype=dtype),
                torch.randn(48, 32, device=device, dtype=dtype),
            )
        )
        for tensor in args:
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)
        with config.patch(
            force_disable_caches=True, post_grad_fusion_options=guarded_options()
        ):
            actual = torch.compile(parallel_linear, fullgraph=True)(*args)
            expected = parallel_linear(*args)
            self.assertEqual(actual, expected, atol=0.02, rtol=0.02)
            weights = tuple(torch.randn_like(t) for t in actual)
            self.assertEqual(
                torch.autograd.grad(actual, args, weights),
                torch.autograd.grad(expected, args, weights),
                atol=0.02,
                rtol=0.02,
            )

    def test_dependent_matmuls_do_not_speculate(self, device):
        def fn(x, *weights):
            outputs = []
            for weight in weights:
                x = F.linear(x, weight)
                outputs.append(x)
            return tuple(outputs)

        torch._dynamo.reset()
        counters.clear()
        x = torch.randn(64, 32, device=device)
        weights = tuple(torch.randn(32, 32, device=device) / 8 for _ in range(5))
        torch._dynamo.mark_dynamic(x, 0)
        with config.patch(
            force_disable_caches=True, post_grad_fusion_options=guarded_options()
        ):
            actual = torch.compile(fn, fullgraph=True)(x, *weights)
        self.assertEqual(actual, fn(x, *weights), rtol=2e-4, atol=2e-4)
        self.assertEqual(counters["inductor"]["guarded_batch_linear_attempts"], 0)

    def test_raw_storage_users_do_not_speculate(self, device):
        def fn(*args):
            outputs = parallel_linear(*args)
            return tuple(
                t.as_strided((t.shape[0], t.shape[1] - 1), t.stride(), 1)
                for t in outputs
            )

        torch._dynamo.reset()
        counters.clear()
        args = tuple(
            t
            for _ in range(5)
            for t in (
                torch.randn(64, 32, device=device),
                torch.randn(48, 32, device=device),
            )
        )
        for tensor in args:
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)
        with config.patch(
            force_disable_caches=True, post_grad_fusion_options=guarded_options()
        ):
            actual = torch.compile(fn, fullgraph=True)(*args)
        self.assertEqual(actual, fn(*args), rtol=2e-4, atol=2e-4)
        self.assertEqual(counters["inductor"]["guarded_batch_linear_attempts"], 0)

    @parametrize("training", [False, True])
    def test_shared_lhs_proved_equal(self, device, training):
        def fn(x, *weights):
            return tuple(F.linear(x, w) for w in weights)

        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        torch._dynamo.reset()
        counters.clear()
        with config.patch(
            force_disable_caches=True,
            post_grad_fusion_options={"batch_linear_post_grad": {}},
        ):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for m in (64, 96, 80):
                x = torch.randn(m, 32, device=device, requires_grad=training)
                weights = tuple(
                    torch.randn(48, 32, device=device, requires_grad=training)
                    for _ in range(5)
                )
                torch._dynamo.mark_dynamic(x, 0)
                actual = compiled(x, *weights)
                expected = fn(x, *weights)
                self.assertEqual(actual, expected, rtol=2e-4, atol=2e-4)
                if training:
                    grads = tuple(torch.randn_like(t) for t in actual)
                    self.assertEqual(
                        torch.autograd.grad(actual, (x, *weights), grads),
                        torch.autograd.grad(expected, (x, *weights), grads),
                        rtol=2e-4,
                        atol=2e-4,
                    )
        self.assertEqual(len(graphs), 1)
        self.assertGreater(counters["inductor"]["batch_linear_post_grad"], 0)
        self.assertEqual(counters["inductor"]["guarded_batch_linear_attempts"], 0)

    @parametrize("training", [False, True])
    @parametrize("strided", [False, True])
    @parametrize("first_case", ["equal", "unequal", "oversize"])
    def test_guarded_fallback_and_working_set(
        self, device, training, strided, first_case
    ):
        torch._dynamo.reset()
        counters.clear()
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        shapes = [
            (64, 32, 48, None),
            (72, 24, 40, "n"),
            (80, 24, 48, "m"),
            (80, 32, 40, "k"),
            (256, 256, 256, None),
            (96, 40, 56, None),
            (64, 32, 48, None),
        ]
        if first_case == "unequal":
            shapes[0], shapes[1] = shapes[1], shapes[0]
        elif first_case == "oversize":
            shapes[0], shapes[4] = shapes[4], shapes[0]
        with config.patch(
            force_disable_caches=True, post_grad_fusion_options=guarded_options()
        ):
            compiled = torch.compile(parallel_linear, backend=backend, fullgraph=True)
            for m, k, n, mismatch in shapes:
                args = []
                for index in range(5):
                    mm = m + (8 if index == 4 and mismatch == "m" else 0)
                    kk = k + (8 if index == 4 and mismatch == "k" else 0)
                    nn = n + (8 if index == 4 and mismatch == "n" else 0)
                    for shape in ((mm, kk), (nn, kk)):
                        t = torch.randn(
                            shape[0], shape[1] * (2 if strided else 1), device=device
                        )
                        if strided:
                            t = t[:, ::2]
                        t.requires_grad_(training)
                        torch._dynamo.mark_dynamic(t, 0)
                        torch._dynamo.mark_dynamic(t, 1)
                        args.append(t)
                expected = parallel_linear(*args)
                codes = []
                with patch.object(GraphLowering, "save_output_code", codes.append):
                    actual = compiled(*args)
                    self.assertEqual(actual, expected, rtol=2e-4, atol=2e-4)
                    if training:
                        weights = tuple(torch.randn_like(t) for t in actual)
                        self.assertEqual(
                            torch.autograd.grad(actual, args, weights),
                            torch.autograd.grad(expected, args, weights),
                            rtol=2e-4,
                            atol=2e-4,
                        )
                if codes and torch.device(device).type == "mps":
                    should_batch = mismatch is None and m != 256
                    self.assertEqual("extern_kernels.bmm(" in codes[0], should_batch)
        self.assertEqual(len(graphs), 2)

    @parametrize(
        "options,expected",
        [
            ({}, False),
            ({"guarded_batching": True}, True),
            ({"guarded_batching": True, "max_guarded_conditions": 1}, False),
            ({"guarded_batching": True, "max_guarded_group_size": 4}, False),
            ({"guarded_batching": True, "max_guarded_group_size": 0}, False),
            ({"guarded_batching": True, "max_guarded_conditions": 0}, False),
            ({"guarded_batching": True, "max_guarded_working_set_bytes": 0}, False),
        ],
    )
    def test_opt_in_and_budgets(self, device, options, expected):
        torch._dynamo.reset()
        counters.clear()
        args = tuple(
            t
            for _ in range(5)
            for t in (
                torch.randn(64, 32, device=device),
                torch.randn(48, 32, device=device),
            )
        )
        for t in args:
            torch._dynamo.mark_dynamic(t, 0)
            torch._dynamo.mark_dynamic(t, 1)
        with config.patch(
            force_disable_caches=True,
            post_grad_fusion_options={"batch_linear_post_grad": options},
        ):
            actual = torch.compile(parallel_linear, fullgraph=True)(*args)
        self.assertEqual(actual, parallel_linear(*args), rtol=2e-4, atol=2e-4)
        self.assertEqual(counters["inductor"]["guarded_batch_linear"], int(expected))
        self.assertLessEqual(counters["inductor"]["guarded_batch_linear_attempts"], 1)

    def test_one_guarded_group_budget(self, device):
        torch._dynamo.reset()
        counters.clear()
        args = tuple(
            t
            for _ in range(10)
            for t in (
                torch.randn(64, 32, device=device),
                torch.randn(48, 32, device=device),
            )
        )
        for t in args:
            torch._dynamo.mark_dynamic(t, 0)
            torch._dynamo.mark_dynamic(t, 1)
        codes = []
        with (
            config.patch(force_disable_caches=True),
            patch.object(GraphLowering, "save_output_code", codes.append),
        ):
            actual = torch.compile(
                parallel_linear,
                fullgraph=True,
                options={
                    "post_grad_fusion_options": guarded_options(
                        max_guarded_group_size=5
                    )
                },
            )(*args)
        self.assertEqual(actual, parallel_linear(*args), rtol=2e-4, atol=2e-4)
        self.assertEqual(counters["inductor"]["guarded_batch_linear_attempts"], 1)
        self.assertEqual(counters["inductor"]["guarded_batch_linear"], 1)
        if torch.device(device).type == "mps":
            self.assertEqual(codes[0].count("extern_kernels.bmm("), 1)
            self.assertEqual(codes[0].count("extern_kernels.mm("), 5)

    def test_known_addmm_and_mixed_dtype(self, device):
        if torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 unsupported")

        def fn(x, y, *weights):
            return tuple(
                torch.addmm(w[0], x if i < 5 else y, w[1])
                for i, w in enumerate(zip(weights[::2], weights[1::2]))
            )

        torch._dynamo.reset()
        counters.clear()
        x = torch.randn(64, 32, device=device)
        y = x.to(torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 0)
        args = tuple(
            t
            for i in range(10)
            for t in (
                torch.randn(
                    48, device=device, dtype=torch.float32 if i < 5 else torch.bfloat16
                ),
                torch.randn(
                    32,
                    48,
                    device=device,
                    dtype=torch.float32 if i < 5 else torch.bfloat16,
                ),
            )
        )
        with config.patch(
            force_disable_caches=True,
            post_grad_fusion_options={"batch_linear_post_grad": {}},
        ):
            actual = torch.compile(fn, fullgraph=True)(x, y, *args)
        self.assertEqual(actual, fn(x, y, *args), rtol=0.02, atol=0.02)
        self.assertEqual(counters["inductor"]["batch_linear_post_grad"], 2)

    @parametrize("m", [0, 1])
    def test_degenerate_shapes_do_not_speculate(self, device, m):
        torch._dynamo.reset()
        counters.clear()
        args = tuple(
            t
            for _ in range(5)
            for t in (
                torch.randn(m, 32, device=device),
                torch.randn(48, 32, device=device),
            )
        )
        for t in args:
            for d, size in enumerate(t.shape):
                if size > 1:
                    torch._dynamo.mark_dynamic(t, d)
        with config.patch(
            force_disable_caches=True, post_grad_fusion_options=guarded_options()
        ):
            actual = torch.compile(parallel_linear, fullgraph=True)(*args)
        self.assertEqual(actual, parallel_linear(*args), rtol=2e-4, atol=2e-4)
        self.assertEqual(counters["inductor"]["guarded_batch_linear"], 0)


instantiate_parametrized_tests(TestBatchShapeRelations)
instantiate_device_type_tests(
    TestSymbolicBatchLinear,
    globals(),
    except_for=("cpu", "xpu", "privateuse1"),
    allow_mps=True,
)

if __name__ == "__main__":
    run_tests()
