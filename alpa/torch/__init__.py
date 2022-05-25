from typing import Any, Callable, Optional
import contextlib

from . import nn
from . import ops
from . import optim
from .ops.mapping import bind_ops

import torch
from torchdistx import deferred_init as torchdistx_deferred_init
import alpa
import numpy as np
from .utils import make_shaped_array_from_pt_tensor

# "local": pure PT eager mode on single GPU, allows print in middle of graph, no dist training
# "dist": graph mode by lowering PT program to JAX, doesn't allow print, dist training
_mode = "local"


def set_mode(mode: str):
    global _mode
    assert mode in ["local", "dist"]
    _mode = mode


def mode():
    return _mode


def manual_seed(seed: int):
    # This sets both `torch.manual_seed` and XLA seed under the hood.
    # TODO: add XLA seed setting 
    torch.manual_seed(seed)


def to_format(format: str, inp: Any):
    # Converts inputs to the format specified by `format` (either "torch" or "alpa").
    assert format in ["torch", "alpa"]
    if isinstance(inp, tuple):
        return tuple([to_format(format, x) for x in inp])
    elif isinstance(inp, list):
        return [to_format(format, x) for x in inp]
    elif isinstance(inp, dict):
        return dict(zip(inp.keys(), [to_format(format, x) for x in inp.values()]))
    elif isinstance(inp, torch.Tensor):
        if format == "alpa":
            if str(inp.device) == "meta":
                return make_shaped_array_from_pt_tensor(inp)
            elif str(inp.device) == "cpu":
                return inp.numpy()
            else:
                # TODO: add support for CUDA input tensor
                raise NotImplementedError(f"PyTorch tensor of device {type(inp.device)} is not supported yet.")
        elif format == "torch":
            return inp
    elif isinstance(inp, alpa.device_mesh.DistributedArray):
        if format == "torch":
            return torch.from_numpy(np.array(inp))
        elif format == "alpa":
            return inp
    else:
        raise NotImplementedError(f"Value of type {type(inp)} is not supported yet.")


def assert_format(format: str, *inputs):
    # Asserts inputs are in the format specified by `format` (either "torch" or "alpa").
    assert format in ["torch", "alpa"]
    for inp in inputs:
        if isinstance(inp, (tuple, list)):
            assert_format(format, *inp)
        elif isinstance(inp, dict):
            assert_format(format, *inp.values())
        else:
            assert (isinstance(inp, torch.Tensor) and format == "torch") or \
            (isinstance(inp, alpa.device_mesh.DistributedArray) and format == "alpa"), \
            f"This input is not of {format} format: {inp}, of type {type(inp)}"


def meta_init(module_fn: Callable[..., torch.nn.Module], *args, **kwargs):
    # We apply both `decompose_ops` and `infer_output_shape` context managers,
    # to both decompose the large PyTorch ops into smaller ones and add meta tensor support to ops as needed.
    with alpa.torch.ops.decompose_ops():
        with alpa.torch.ops.infer_output_shape():
            pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args, **kwargs)
    pt_module = pt_module.to(device="meta")
    return pt_module


def materialize_module(
    module: torch.nn.Module,
    buffers_only: bool = False,
    check_fn: Optional[Callable[[torch.nn.Module], bool]] = None,
):
    return torchdistx_deferred_init.materialize_module(module, buffers_only=buffers_only, check_fn=check_fn)


@contextlib.contextmanager
def run_init():
    _prev_use_dummy_value_for_benchmarking = alpa.global_env.global_config.use_dummy_value_for_benchmarking
    alpa.global_env.global_config.use_dummy_value_for_benchmarking = True
    yield
    alpa.global_env.global_config.use_dummy_value_for_benchmarking = _prev_use_dummy_value_for_benchmarking
