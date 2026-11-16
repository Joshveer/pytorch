# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


REDUCTIONS = ("amax", "amin", "prod", "var", "std", "max", "min", "var_mean")


def reduction_epilogue(x, y, kind, dims):
    if kind in ("var", "std"):
        reduced = getattr(x, kind)(dim=dims, keepdim=True, correction=0)
    elif kind in ("max", "min"):
        reduced = getattr(x, kind)(dim=dims[-1], keepdim=True).values
    elif kind == "prod":
        reduced = x.prod(dim=dims[-1], keepdim=True)
    elif kind == "var_mean":
        variance, mean = torch.var_mean(x, dim=dims, keepdim=True, correction=0)
        reduced = variance + mean
    else:
        reduced = getattr(x, kind)(dim=dims, keepdim=True)
    return y * (reduced + 0.1)


class TestGeneralShapeFusion(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def make_inputs(self, device, shapes, training=False, strided=False, dynamic=True):
        inputs = []
        for shape in shapes:
            storage_shape = (*shape[:-1], shape[-1] * (2 if strided else 1))
            # Keep products well-scaled and extrema distinct for gradient checks.
            t = 0.99 + 0.02 * torch.rand(storage_shape, device=device)
            if strided:
                t = t[..., ::2]
            t.requires_grad_(training)
            if dynamic:
                for dim in range(t.ndim):
                    if t.shape[dim] > 1:
                        torch._dynamo.mark_dynamic(t, dim)
            inputs.append(t)
        return inputs

    def check_result(self, fn, compiled, inputs, training):
        actual = compiled(*inputs)
        expected = fn(*inputs)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        if training:
            weight = torch.randn_like(actual)
            self.assertEqual(
                torch.autograd.grad(actual, inputs, weight),
                torch.autograd.grad(expected, inputs, weight),
                atol=3e-4,
                rtol=3e-4,
            )

    @parametrize("kind", REDUCTIONS)
    def test_static_dynamic_fusion_gap(self, device, kind):
        def fn(x, y):
            return reduction_epilogue(x, y, kind, (-1,))

        counts = []
        for dynamic, enabled in ((False, False), (True, False), (True, True)):
            torch._dynamo.reset()
            inputs = self.make_inputs(device, [(17, 64), (17, 64)], dynamic=dynamic)
            codes = []
            with (
                config.patch(force_disable_caches=True, guarded_shape_fusion=enabled),
                patch.object(GraphLowering, "save_output_code", codes.append),
            ):
                compiled = torch.compile(fn, fullgraph=True, dynamic=dynamic)
                self.check_result(fn, compiled, inputs, False)
            counts.append(sum(c.count("= async_compile.metal(") for c in codes))
        if torch.device(device).type == "mps":
            self.assertEqual(counts, [1, 2, 1])

    @parametrize("kind", REDUCTIONS)
    @parametrize("training", [False, True])
    @parametrize("strided", [False, True])
    @parametrize("unequal_first", [False, True])
    def test_equal_and_unequal_stream(
        self, device, kind, training, strided, unequal_first
    ):
        def fn(x, y):
            return reduction_epilogue(x, y, kind, (-1,))

        shapes = [
            ((17, 64), (17, 64)),
            ((17, 48), (17, 80)),
            ((19, 96), (19, 96)),
            ((19, 72), (19, 56)),
            ((17, 64), (17, 64)),
        ]
        if unequal_first:
            shapes[0], shapes[1] = shapes[1], shapes[0]
        self.check_stream(device, fn, shapes, training, strided)

    @parametrize("kind", ["amax", "var", "var_mean"])
    @parametrize("training", [False, True])
    @parametrize("unequal_first", [False, True])
    def test_joint_suffix_guard(self, device, kind, training, unequal_first):
        def fn(x, y):
            return reduction_epilogue(x, y, kind, (-2, -1))

        # Equal element counts with unequal axes must not be treated as equality.
        shapes = [
            ((7, 4, 16), (7, 4, 16)),
            ((7, 4, 16), (7, 8, 8)),
            ((9, 6, 24), (9, 6, 24)),
            ((9, 6, 24), (9, 8, 24)),
            ((9, 6, 24), (9, 6, 32)),
        ]
        if unequal_first:
            shapes[0], shapes[1] = shapes[1], shapes[0]
        self.check_stream(device, fn, shapes, training, False)

    def check_stream(self, device, fn, shapes, training, strided):
        torch._dynamo.reset()
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        with config.patch(force_disable_caches=True, guarded_shape_fusion=True):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for pair in shapes:
                inputs = self.make_inputs(device, pair, training, strided)
                codes = []
                with patch.object(GraphLowering, "save_output_code", codes.append):
                    self.check_result(fn, compiled, inputs, training)
                if codes and torch.device(device).type == "mps":
                    self.assertEqual(
                        codes[0].count("= async_compile.metal("),
                        1 if pair[0] == pair[1] else 2,
                    )
            self.assertEqual(len(graphs), 2)

    @parametrize("kind", ["amax", "prod", "var"])
    @parametrize("shapes", [((0, 8), (0, 8)), ((3, 1), (3, 8)), ((3, 8), (3, 1))])
    def test_degenerate_dimensions(self, device, kind, shapes):
        def fn(x, y):
            return reduction_epilogue(x, y, kind, (-1,))

        torch._dynamo.reset()
        inputs = self.make_inputs(device, shapes, training=True)
        with config.patch(force_disable_caches=True, guarded_shape_fusion=True):
            self.check_result(fn, torch.compile(fn, fullgraph=True), inputs, True)

    @parametrize("kind", ["leading_axis", "all_axes", "indices", "slice", "budget"])
    def test_unsupported_patterns_do_not_specialize(self, device, kind):
        def fn(x, y):
            if kind == "leading_axis":
                return y * x.amax(0, keepdim=True)
            if kind == "all_axes":
                return y * x.amax()
            if kind == "indices":
                return y * x.argmax(-1, keepdim=True)
            if kind == "slice":
                return y * x[:, :1]
            return y * x.amax(tuple(range(1, x.ndim)), keepdim=True)

        if kind == "leading_axis":
            pairs = [((32, 7), (32, 7)), ((48, 9), (64, 9)), ((40, 11), (40, 11))]
        elif kind == "budget":
            pairs = [
                ((7, 2, 3, 4, 5, 6), (7, 2, 3, 4, 5, 6)),
                ((9, 3, 4, 5, 6, 7), (9, 4, 3, 6, 7, 5)),
                ((7, 3, 4, 5, 6, 7), (7, 3, 4, 5, 6, 7)),
            ]
        else:
            pairs = [((7, 32), (7, 32)), ((9, 48), (9, 64)), ((11, 40), (11, 40))]
        torch._dynamo.reset()
        graphs = []

        def backend(gm, inputs):
            graphs.append(gm)
            return torch._inductor.compile(gm, inputs)

        with config.patch(force_disable_caches=True, guarded_shape_fusion=True):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for pair in pairs:
                inputs = self.make_inputs(device, pair)
                self.check_result(fn, compiled, inputs, False)
            self.assertEqual(len(graphs), 1)

    def test_tuple_indices_remain_correct(self, device):
        def fn(x, y):
            value, index = torch.max(x, dim=-1, keepdim=True)
            return y * value, index

        torch._dynamo.reset()
        with config.patch(force_disable_caches=True, guarded_shape_fusion=True):
            compiled = torch.compile(fn, fullgraph=True)
            for left, right in ((64, 64), (48, 80), (96, 96)):
                inputs = self.make_inputs(device, [(17, left), (17, right)])
                self.assertEqual(compiled(*inputs), fn(*inputs))


instantiate_device_type_tests(
    TestGeneralShapeFusion,
    globals(),
    except_for=("cpu", "xpu", "privateuse1"),
    allow_mps=True,
)

if __name__ == "__main__":
    run_tests()
