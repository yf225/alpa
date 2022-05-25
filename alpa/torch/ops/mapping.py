import contextlib
import math
import functools

import torch
import functorch
import jax
import jax.numpy as jnp
from jax import lax
import alpa
from alpa.pipeline_parallel.primitive_def import mark_gradient
from typing import Optional, Any
from alpa.torch.utils import torch_to_numpy_dtype_dict
from alpa.device_mesh import DistributedArray
from jax.interpreters import pxla


def torch__unsafe_view(tensor, shape):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return lax.reshape(tensor, shape)

def torch_add(input, other):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.add(input, other)

def torch_bmm(input, mat2):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return lax.batch_matmul(input, mat2)

def torch_cat(tensors, dim=0):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return lax.concatenate(tensors, dim)

def torch_clone(input, memory_format=torch.preserve_format):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.array(input, dtype=input.dtype, copy=True, order='K')

def torch_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    # References:
    # - torch-xla impl and haiku / flax impl
    # - https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/convolutions.ipynb
    conv_out = lax.conv_general_dilated(
        input, weight, stride, [(x, x) for x in padding],
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=lax.conv_dimension_numbers(
            input.shape,
            weight.shape,
            ('NCHW', 'OIHW', 'NCHW'),  # TODO: parameterize this! don't assume NCHW format.
        ),
        feature_group_count=groups,
        batch_group_count=1,
    )
    if bias is not None:
        bias_reshaped = bias.reshape(1, bias.shape[0], 1, 1)
        bias_reshaped = jnp.broadcast_to(bias_reshaped, [conv_out.shape[0], bias.shape[0], conv_out.shape[2], conv_out.shape[3]])
        return conv_out + bias_reshaped
    else:
        return conv_out

def torch_div(input, other, rounding_mode=None):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    if rounding_mode is None:
        return jnp.true_divide(input, other)
    elif rounding.mode() == "trunc":
        return jnp.trunc(jnp.true_divide(input, other))
    elif rounding.mode() == "floor":
        return jnp.floor_divide(input, other)
    else:
        raise NotImplementedError(str(rounding_mode) + "is not supported")

def torch_exp(input):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.exp(input)

def torch_expand(tensor, sizes):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    computed_sizes = list(sizes)
    for dim, size in enumerate(sizes):
        if size == -1:
            computed_sizes[dim] = tensor.shape[dim]
    return lax.broadcast_in_dim(tensor, computed_sizes, list(range(len(tensor.shape))))

def torch_gelu(input, approximate=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    # TODO: use approximate=True or not?
    return jax.nn.gelu(input)

def torch_matmul(input, other):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.matmul(input, other)

def torch_max(input, dim=None, keepdim=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.max(input, axis=dim, keepdims=keepdim)

def torch_mean(input, dim=None, keepdim=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.mean(input, axis=dim, keepdims=keepdim)

def torch_mm(input, mat2):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.matmul(input, mat2)

def torch_mul(x1, x2):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.multiply(x1, x2)

def torch_permute(input, dims):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.transpose(input, dims)

def torch_pow(input, exponent):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.power(input, exponent)

def torch_select(input, dim, index):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    # TODO: likely inefficient. What's the better way?
    return lax.slice_in_dim(input, index, index+1, stride=1, axis=dim)[0]

def torch_slice(input, dim, start, end, step=1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    if end > input.shape[dim]:
        end = input.shape[dim]
    return lax.slice_in_dim(input, start, end, stride=step, axis=dim)

def torch_split(tensor, split_size_or_sections, dim=0):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    if type(split_size_or_sections) == int:
        split_size = split_size_or_sections
        sections = list(range(split_size, tensor.shape[dim], split_size))
    else:
        assert type(split_size_or_sections) == list
        sections = split_size_or_sections
    return jnp.split(tensor, sections, axis=dim)

def torch_sqrt(input):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.sqrt(input)

def torch_sub(input, other, alpha=1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return input - alpha * other

def torch_sum(input, dim, keepdim=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.sum(input, axis=dim, keepdims=keepdim)

def torch_t(input):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.transpose(input)

def torch_transpose(input, dim0, dim1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.swapaxes(input, dim0, dim1)

def torch_unbind(input, dim=0):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    ret = []
    for index in range(input.shape[dim]):
        # TODO: likely inefficient. What's the better way?
        ret.append(lax.slice_in_dim(input, index, index+1, stride=1, axis=dim)[0])
    return tuple(ret)

def torch_view(tensor, shape):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return lax.reshape(tensor, shape)

def torch_softmax(x, dim):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    x_max = jnp.max(x, axis=dim, keepdims=True)[0]
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=dim, keepdims=True)

def torch_nn_functional_softmax(x, dim):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return torch_softmax(x=x, dim=dim)

def torch_dropout(input, p=0.5, training=True, inplace=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    assert not inplace, "Inplace dropout is not supported"
    if p == 0.:
      return input
    if training:
        # Copied from flax.linen.Dropout impl
        keep_prob = 1. - p
        # NOTE: pass None for rng, since Alpa ignores it anyway.
        mask = jax.random.bernoulli(None, p=keep_prob, shape=input.shape)
        return lax.select(mask, input, jnp.zeros_like(input))
    else:
        return input

def torch_nn_functional_dropout(input, p=0.5, training=True, inplace=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return torch_dropout(input, p=p, training=training, inplace=inplace)

def torch_abs(input):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jnp.absolute(input)

def torch__normalize(x, mean, var, weight, bias, reduction_axes, feature_axes, eps):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
    y = x - mean
    mul = lax.rsqrt(var + eps)
    if weight is not None:
        mul *= weight.reshape(feature_shape)
    y *= mul
    if bias is not None:
        y += bias.reshape(feature_shape)
    return jnp.asarray(y, x.dtype)

def torch_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    # Ref: https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.BatchNorm.html
    def _abs_sq(x):
        """Computes the elementwise square of the absolute value |x|^2."""
        if jnp.iscomplexobj(x):
            return lax.square(lax.real(x)) + lax.square(lax.imag(x))
        else:
            return lax.square(x)

    def _compute_stats(x, axes,
                       axis_name: Optional[str] = None,
                       axis_index_groups: Any = None):
        # promote x to at least float32, this avoids half precision computation
        # but preserves double or complex floating points
        x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
        mean = jnp.mean(x, axes)
        mean2 = jnp.mean(_abs_sq(x), axes)
        if axis_name is not None:
            concatenated_mean = jnp.concatenate([mean, mean2])
            mean, mean2 = jnp.split(
                lax.pmean(
                    concatenated_mean,
                    axis_name=axis_name,
                    axis_index_groups=axis_index_groups), 2)
        # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
        # to floating point round-off errors.
        var = jnp.maximum(0., mean2 - _abs_sq(mean))
        return mean, var

    feature_axes = [1]  # Expect (N, C, ...) shape
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    if not training:
        mean, var = running_mean, running_var
    else:
        running_mean = jnp.zeros(feature_shape, jnp.float32)
        running_var = jnp.ones(feature_shape, jnp.float32)
        mean, var = _compute_stats(x, reduction_axes)

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

    out = _normalize(x, mean, var, weight, bias, reduction_axes, feature_axes, eps)

    return out, running_mean, running_var

def torch_relu(input, inplace=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return jax.nn.relu(input)

def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max)
    if dim < 0:
        dim += dim_post_expr
    return dim

def torch_flatten(input, start_dim=0, end_dim=-1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    input_shape = input.shape
    start_dim = maybe_wrap_dim(start_dim, len(input_shape))
    end_dim = maybe_wrap_dim(end_dim, len(input_shape))
    assert start_dim <= end_dim
    if start_dim == end_dim:
        return input
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input_shape[i]
    shape = []
    for i in range(start_dim):
        shape.append(input_shape[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input_shape)):
        shape.append(input_shape[i])
    return torch_view(input, shape)

# TODO: dedup this with impl in decomposition
def torch_addmm(self, mat1, mat2, beta=1, alpha=1):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    out = alpha * torch.matmul(mat1, mat2)
    if beta == 0:
        return out
    return beta * self + out

def torch_nn_functional_batch_norm(
    x: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    return torch_batch_norm(
        x=x,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=training,
        momentum=momentum,
        eps=eps,
    )

def torch_nn_functional_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    # TODO: add handling for `size_average` / `reduce` / `reduction`
    return jnp.mean((input - target) ** 2)

# TODO: dedup this with decomposition.py
def torch_nn_functional_linear(input, weight, bias=None):
    output = torch.matmul(input, torch.t(weight))
    if bias is not None:
        output = output + bias
    return output

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if len(tensor.shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def torch_nn_init_xavier_uniform(array, gain: float = 1.):
    if alpa.torch.mode() == "local":
        # TODO: should materialize and then call original impl instead
        raise NotImplementedError
    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return DistributedArray.init_like(
        array,
        jax.random.uniform,
        key=None,  # TODO: is random key already handled by Alpa?
        minval=-a,
        maxval=a,
    )


def torch_nn_init_normal(array, mean: float = 0., std: float = 1.):
    if alpa.torch.mode() == "local":
        # TODO: should materialize and then call original impl instead
        raise NotImplementedError
    return DistributedArray.init_like(
        array,
        jax.random.normal,
        key=None,  # TODO: is random key already handled by Alpa?
        mu=mean,
        sigma=std,
    )


def torch_nn_init_zeros(array):
    if alpa.torch.mode() == "local":
        return torch.zeros(array.shape, device="cpu")
    else:
        # return array.init_with_func(jax.numpy.zeros)
        return DistributedArray.init_like(array, jax.numpy.zeros)


def functorch_grad_and_value(func, argnums=0, has_aux=False):
    if alpa.torch.mode() == "local":
        raise NotImplementedError
    """The same implementation as alpa.value_and_grad, but puts grad first and value second in output.
    """
    def ret(*call_args, **call_kwargs):
        value_and_grad_func = alpa.api.value_and_grad(func, argnums=argnums, has_aux=has_aux)
        val, grad = value_and_grad_func(*call_args, **call_kwargs)
        return mark_gradient((grad, val))

    return ret

# PyTorch .detach() is equivalent to JAX lax.stop_gradient(): https://github.com/google/jax/issues/2025
# PyTorch .view() is equivalent to JAX lax.reshape(): https://jax.readthedocs.io/en/latest/_autosummary/lax.reshape.html


op_orig_impl_dict = {}
op_patch_list = [
    # (torch, "_unsafe_view", torch__unsafe_view),
    (torch, "add", torch_add),
    (torch, "bmm", torch_bmm),
    (torch, "cat", torch_cat),
    (torch, "clone", torch_clone),
    (torch, "conv2d", torch_conv2d),
    (torch, "div", torch_div),
    (torch, "exp", torch_exp),
    (torch, "expand", torch_expand),
    # (torch, "gelu", torch_gelu),
    (torch, "matmul", torch_matmul),
    (torch, "max", torch_max),
    (torch, "mean", torch_mean),
    (torch, "mm", torch_mm),
    (torch, "mul", torch_mul),
    (torch, "permute", torch_permute),
    (torch, "pow", torch_pow),
    (torch, "select", torch_select),
    # (torch, "slice", torch_slice),
    (torch, "softmax", torch_softmax),
    (torch, "split", torch_split),
    (torch, "sqrt", torch_sqrt),
    (torch, "sub", torch_sub),
    (torch, "sum", torch_sum),
    (torch, "t", torch_t),
    (torch, "transpose", torch_transpose),
    (torch, "unbind", torch_unbind),
    (torch, "view", torch_view),
    (torch, "dropout", torch_dropout),
    (torch, "abs", torch_abs),
    (torch, "relu", torch_relu),
    (torch, "flatten", torch_flatten),
    (torch, "addmm", torch_addmm),
    (torch.nn.functional, "softmax", torch_nn_functional_softmax),
    (torch.nn.functional, "dropout", torch_nn_functional_dropout),
    (torch.nn.functional, "batch_norm", torch_nn_functional_batch_norm),
    (torch.nn.functional, "mse_loss", torch_nn_functional_mse_loss),
    (torch.nn.functional, "linear", torch_nn_functional_linear),
    (torch.nn.init, "xavier_uniform", torch_nn_init_xavier_uniform),
    (torch.nn.init, "normal", torch_nn_init_normal),
    (torch.nn.init, "zeros", torch_nn_init_zeros),
    # (torch.nn.init, "zeros", torch_nn_init_zeros),
    (functorch, "grad_and_value", functorch_grad_and_value),
    # TODO: add hard error for in-place ops
]


def patch_ops():
    for python_module, op_name, new_impl in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        op_orig_impl_dict[f"{python_module_fqn}.{op_name}"] = getattr(python_module, op_name, None)
        def patch_if_implemented(python_module_fqn, op_name, new_impl, *args, **kwargs):
            try:
                return new_impl(*args, **kwargs)
            except NotImplementedError:
                return op_orig_impl_dict[f"{python_module_fqn}.{op_name}"](*args, **kwargs)
        setattr(python_module, op_name, functools.partial(patch_if_implemented, python_module_fqn, op_name, new_impl))


def unpatch_ops():
    for python_module, op_name, _ in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        op_orig_impl = op_orig_impl_dict[f"{python_module_fqn}.{op_name}"]
        if op_orig_impl is not None:
            setattr(python_module, op_name, op_orig_impl)


@contextlib.contextmanager
def bind_ops(enabled: bool = True):
    """
    Context manager within which many PyTorch ops are monkey-patched to also accept Alpa arrays.
    """
    if enabled:
        patch_ops()
    yield
    if enabled:
        unpatch_ops()
