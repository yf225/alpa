try:
    import torch
except ImportError as e:
    print(
        """
Attempted to use Alpa-PyTorch frontend, but PyTorch is not installed.

Please follow instructions at docs/torch_build_instructions.txt to install PyTorch and related dependencies.
"""
    )
    raise e

from typing import Any, Callable

from torchdistx import deferred_init as torchdistx_deferred_init
import numpy as np
import alpa
from alpa.device_mesh import DistributedArray
from alpa.torch.ops.utils import is_torch_tensor_type

from . import nn, ops, optim
from .nn import functionalize
from .ops.mapping import bind_ops
from .utils import array_init_like, make_shaped_array_from_pt_tensor

# "local": pure PT eager mode on single GPU, allows print in middle of graph, no dist training
# "dist": graph mode by lowering PT program to JAX, doesn't allow print, dist training
_mode = "local"


# If True, prints verbose log for debugging.
debug = False


def set_mode(new_mode: str):
    global _mode
    assert new_mode in ["local", "dist"]
    _mode = new_mode


def mode():
    return _mode


def manual_seed(seed: int):
    # This sets both `torch.manual_seed` and XLA seed under the hood.
    # TODO: add XLA seed setting
    torch.manual_seed(seed)


def to_format(target_format: str, inp: Any):
    # Converts inputs to the format specified by `target_format` (either "torch" or "alpa").
    assert target_format in ["torch", "alpa"]
    ret = None
    if isinstance(inp, tuple):
        ret = tuple(to_format(target_format, x) for x in inp)
    elif isinstance(inp, list):
        ret = [to_format(target_format, x) for x in inp]
    elif isinstance(inp, dict):
        ret = dict(zip(inp.keys(), [to_format(target_format, x) for x in inp.values()]))
    elif isinstance(inp, torch.Tensor):
        if target_format == "alpa":
            if str(inp.device) == "meta":
                ret = make_shaped_array_from_pt_tensor(inp)
            elif str(inp.device) == "cpu":
                ret = inp.numpy()
            else:
                # TODO: add support for CUDA input tensor
                raise NotImplementedError(f"PyTorch tensor of device {type(inp.device)} is not supported yet.")
        elif target_format == "torch":
            ret = inp
    elif isinstance(inp, alpa.device_mesh.DistributedArray):
        if target_format == "torch":
            ret = torch.from_numpy(np.array(inp))
        elif target_format == "alpa":
            ret = inp
    if ret is not None:
        return ret
    else:
        raise NotImplementedError(f"Value of type {type(inp)} is not supported yet.")


def assert_format(target_format: str, *inputs):
    # Asserts inputs are in the format specified by `target_format` (either "torch" or "alpa").
    assert target_format in ["torch", "alpa"]
    for inp in inputs:
        if isinstance(inp, (tuple, list)):
            assert_format(target_format, *inp)
        elif isinstance(inp, dict):
            assert_format(target_format, *inp.values())
        else:
            assert (isinstance(inp, torch.Tensor) and target_format == "torch") or (
                isinstance(inp, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray))
                and target_format == "alpa"
            ), f"This input is not of {target_format} format: {inp}, of type {type(inp)}"


def meta_init(module_fn: Callable[..., torch.nn.Module], *args, **kwargs):
    pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args, **kwargs)
    pt_module = pt_module.to(device="meta")
    return pt_module


def grad_and_value(func, argnums=0, has_aux=False):
    if _mode == "local":
        # pylint: disable=import-outside-toplevel
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

    return array_init_like(x, gen_zeros,)
