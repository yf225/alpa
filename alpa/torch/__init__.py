from typing import Any, Callable, Optional
import contextlib

from . import nn
from . import ops
from . import optim
from .ops.mapping import bind_ops

import torch
from torchdistx import deferred_init as torchdistx_deferred_init
import alpa
import alpa.torch as atorch
from alpa.torch.ops.utils import is_torch_tensor_type
from alpa.device_mesh import DistributedArray
import jax
import numpy as np
from .utils import make_shaped_array_from_pt_tensor
from .nn import functionalize
from .utils import array_init_like

# "local": pure PT eager mode on single GPU, allows print in middle of graph, no dist training
# "dist": graph mode by lowering PT program to JAX, doesn't allow print, dist training
_mode = "local"


# If True, prints verbose log for debugging.
debug = False


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
            (isinstance(inp, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray)) and format == "alpa"), \
            f"This input is not of {format} format: {inp}, of type {type(inp)}"


def meta_init(module_fn: Callable[..., torch.nn.Module], *args, **kwargs):
    # # We apply both `decompose_ops` and `infer_output_shape` context managers,
    # # to both decompose the large PyTorch ops into smaller ones and add meta tensor support to ops as needed.
    # with alpa.torch.ops.decompose_ops():
    #     with alpa.torch.ops.infer_output_shape():
    #         pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args, **kwargs)
    pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args, **kwargs)
    pt_module = pt_module.to(device="meta")
    return pt_module


def grad_and_value(func, argnums=0, has_aux=False):
    if atorch.mode() == "local":
        import functorch
        return functorch.grad_and_value(func, argnums=argnums, has_aux=has_aux)
    else:
        return alpa.grad_and_value(func, argnums=argnums, has_aux=has_aux)


def zeros_like(x):
    if is_torch_tensor_type(x):
        return torch.zeros(x.shape, dtype=x.dtype, layout=x.layout, device="cpu", requires_grad=x.requires_grad)
    def gen_zeros(shape, dtype, **kwargs):
        tensor = torch.zeros(shape, dtype=dtype)
        return tensor.numpy()
    return array_init_like(
        x,
        gen_zeros,
    )
