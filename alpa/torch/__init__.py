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

from typing import Any, Callable, Union, Tuple
from functools import partial, wraps

from torchdistx import deferred_init as torchdistx_deferred_init
import numpy as np
import alpa
from alpa.device_mesh import DistributedArray
from alpa.torch import optim
from alpa.torch.nn import functionalize
from alpa.torch.ops.mapping import bind_ops
from alpa.torch.ops.utils import is_torch_tensor_type
from alpa.torch.utils import array_init_like, make_shaped_array_from_pt_tensor

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
    # Converts inputs to the format specified by `target_format` (either "local" or "dist").
    assert target_format in ["local", "dist"]
    ret = None
    if isinstance(inp, tuple):
        ret = tuple(to_format(target_format, x) for x in inp)
    elif isinstance(inp, list):
        ret = [to_format(target_format, x) for x in inp]
    elif isinstance(inp, dict):
        ret = dict(zip(inp.keys(), [to_format(target_format, x) for x in inp.values()]))
    elif isinstance(inp, torch.Tensor):
        if target_format == "dist":
            if str(inp.device) == "meta":
                ret = make_shaped_array_from_pt_tensor(inp)
            elif str(inp.device) == "cpu":
                ret = inp.numpy()
            else:
                # TODO: add support for CUDA input tensor
                raise NotImplementedError(f"PyTorch tensor of device {type(inp.device)} is not supported yet.")
        elif target_format == "local":
            ret = inp
    elif isinstance(inp, alpa.device_mesh.DistributedArray):
        if target_format == "local":
            ret = torch.from_numpy(np.array(inp))
        elif target_format == "dist":
            ret = inp
    if ret is not None:
        return ret
    else:
        raise NotImplementedError(f"Value of type {type(inp)} is not supported yet.")


def assert_format(target_format: str, *inputs):
    # Asserts inputs are in the format specified by `target_format` (either "local" or "dist").
    assert target_format in ["local", "dist"]
    for inp in inputs:
        if isinstance(inp, (tuple, list)):
            assert_format(target_format, *inp)
        elif isinstance(inp, dict):
            assert_format(target_format, *inp.values())
        else:
            assert (isinstance(inp, torch.Tensor) and target_format == "local") or (
                isinstance(inp, (alpa.device_mesh.DistributedArray, alpa.device_mesh.ReplicatedDistributedArray))
                and target_format == "dist"
            ), f"This input is not of {target_format} format: {inp}, of type {type(inp)}"


def meta_init(module_fn: Callable[..., torch.nn.Module], *args, **kwargs):
    pt_module = torchdistx_deferred_init.deferred_init(module_fn, *args, **kwargs)
    pt_module = pt_module.to(device="meta")
    return pt_module


def functorch_value_and_grad(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0, has_aux: bool = False) -> Callable:
    """
    The same implementation as functorch.grad_and_value, but puts value first and grad second in output.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from functorch._C import _grad_increment_nesting, _grad_decrement_nesting
        from functorch._src.eager_transforms import _wrap_all_tensors, _slice_argnums, _create_differentiable, \
            _as_tuple, _autograd_grad, _undo_create_differentiable
        from functorch._src.pytree_hacks import tree_map_
        from torch.utils._pytree import tree_flatten, tree_unflatten
        level = _grad_increment_nesting()
        try:
            output, aux, grad_input = None, None, None
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            "value_and_grad(f)(*args): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    output, aux = output

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError('value_and_grad(f)(*args): Expected f(*args) '
                                       f'to return a Tensor, got {type(output)}')
                if output.dim() != 0:
                    raise RuntimeError('value_and_grad(f)(*args): Expected f(*args) '
                                       'to return a scalar Tensor, got tensor with '
                                       f'{output.dim()} dims. Maybe you wanted to '
                                       'use the vjp or jacrev APIs instead?')

                flat_diff_args, spec = tree_flatten(diff_args)

                # NB: need create_graph so that backward pass isn't run in no_grad mode
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(flat_outputs, flat_diff_args, create_graph=True)
                grad_input = tree_unflatten(flat_grad_input, spec)

                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if aux is not None:
                    aux = _undo_create_differentiable(aux, level)

            if has_aux:
                return (output, aux), grad_input
            return output, grad_input
        finally:
            _grad_decrement_nesting()
    return wrapper


def value_and_grad(func, argnums=0, has_aux=False):
    if _mode == "local":
        return functorch_value_and_grad(func, argnums=argnums, has_aux=has_aux)
    else:
        return alpa.value_and_grad(func, argnums=argnums, has_aux=has_aux)


def _zeros_like(x):
    if is_torch_tensor_type(x):
        return torch.zeros(x.shape, dtype=x.dtype, layout=x.layout, device="cpu", requires_grad=x.requires_grad)

    def gen_zeros(shape, dtype, **kwargs):
        tensor = torch.zeros(shape, dtype=dtype)
        return tensor.numpy()

    return array_init_like(x, gen_zeros,)


def materialize(pt_module, name_map, params, bufs):
    for k, p in pt_module.named_parameters():
        params[name_map[f"{k}"]] = _zeros_like(params[name_map[f"{k}"]])
    for k, b in pt_module.named_buffers():
        bufs[name_map[f"{k}"]] = _zeros_like(bufs[name_map[f"{k}"]])
    return params, bufs


