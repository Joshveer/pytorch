"""Validate the project patch on an installed PyTorch without a source build.

Only the edited simplifier, new pass, config entry, and post-grad call site are
overlaid in this process. This is not validation of the entire source checkout.
Run from outside the checkout, or select the installed torch with PYTHONPATH.
"""

import ast
import importlib.util
import inspect
import runpy
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from torch._inductor import config, sizevars
from torch._inductor.fx_passes import post_grad
from torch.utils._config_module import _Config, _ConfigEntry, ConfigModule


def main():
    root = Path(__file__).resolve().parents[2]
    name = "torch._inductor.fx_passes.shape_relation_fusion"
    path = root / "torch/_inductor/fx_passes/shape_relation_fusion.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    source = root / "torch/_inductor/sizevars.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SizeVarAllocator"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "simplify"
    )
    namespace = dict(vars(sizevars))
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    # Insert the exact source call site into this runtime's post-grad function.
    # Loading the whole newer module would mix unrelated compiler changes.
    source = root / "torch/_inductor/fx_passes/post_grad.py"
    tree = ast.parse(source.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "post_grad_passes"
    )
    hook = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "config.guarded_shape_fusion"
    )
    tree = ast.parse(inspect.getsource(post_grad.post_grad_passes))
    current_fn = tree.body[0]
    if not isinstance(current_fn, ast.FunctionDef):
        raise TypeError("Expected a post_grad_passes function")
    body = current_fn.body
    index = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "config.dce"
    )
    body.insert(index + 1, hook)
    post_namespace = dict(vars(post_grad))
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), post_namespace)
    cast(ConfigModule, config)._config["guarded_shape_fusion"] = _ConfigEntry(
        _Config(default=True), "guarded_shape_fusion"
    )

    print(
        f"Installed runtime: {torch.__version__}, {torch.version.git_version}, {torch.__file__}",
        flush=True,
    )
    with (
        patch.object(sizevars.SizeVarAllocator, "simplify", namespace["simplify"]),
        patch.object(post_grad, "post_grad_passes", post_namespace["post_grad_passes"]),
    ):
        runpy.run_path(
            str(root / "test/inductor/test_shape_relation_fusion.py"),
            run_name="__main__",
        )


if __name__ == "__main__":
    main()
