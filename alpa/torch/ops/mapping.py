import contextlib
import math
import functools

import torch
import functorch
import jax
import jax.numpy as jnp
from jax import lax
import alpa
import alpa.torch as atorch
from alpa.pipeline_parallel.primitive_def import mark_gradient
from typing import Optional, Any
from alpa.torch.utils import torch_to_numpy_dtype_dict
from alpa.device_mesh import DistributedArray
from jax.interpreters import pxla
from alpa.torch.ops.utils import infer_size, is_torch_tensor_type


def torch_add(x, other):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.add"](x, other)
    return jnp.add(x, other)

def torch_bmm(x, mat2):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.bmm"](x, mat2)
    return lax.batch_matmul(x, mat2)

def torch_cat(tensors, dim=0):
    if is_torch_tensor_type(tensors[0]):
        return op_orig_impl_dict["torch.cat"](tensors, dim=dim)
    return lax.concatenate(tensors, dim)

def torch_clone(x, memory_format=torch.preserve_format):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.clone"](x, memory_format=memory_format)
    return jnp.array(x, dtype=x.dtype, copy=True, order='K')

def torch_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.conv2d"](x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # References:
    # - torch-xla impl and haiku / flax impl
    # - https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/convolutions.ipynb
    conv_out = lax.conv_general_dilated(
        x, weight, stride, [(x, x) for x in padding],
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=lax.conv_dimension_numbers(
            x.shape,
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

def torch_div(x, other, rounding_mode=None):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.div"](x, other, rounding_mode=rounding_mode)
    if rounding_mode is None:
        return jnp.true_divide(x, other)
    elif rounding.mode() == "trunc":
        return jnp.trunc(jnp.true_divide(x, other))
    elif rounding.mode() == "floor":
        return jnp.floor_divide(x, other)

def torch_exp(x):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.exp"](x)
    return jnp.exp(x)

def torch_expand(x, sizes):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.expand"](x, sizes)
    computed_sizes = list(sizes)
    for dim, size in enumerate(sizes):
        if size == -1:
            computed_sizes[dim] = x.shape[dim]
    return lax.broadcast_in_dim(x, computed_sizes, list(range(len(x.shape))))

def torch_gelu(x, approximate=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.gelu"](x, approximate=approximate)
    # TODO: use approximate=True or not?
    return jax.nn.gelu(x)

def torch_matmul(x, other):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.matmul"](x, other)
    return jnp.matmul(x, other)

def torch_max(x, dim=None, keepdim=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.max"](x, dim=dim, keepdim=keepdim)
    return jnp.max(x, axis=dim, keepdims=keepdim)

def torch_mean(x, dim=None, keepdim=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.mean"](x, dim=dim, keepdim=keepdim)
    return jnp.mean(x, axis=dim, keepdims=keepdim)

def torch_mm(x, mat2):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.mm"](x, mat2)
    return jnp.matmul(x, mat2)

def torch_mul(x1, x2):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.mul"](x1, x2)
    return jnp.multiply(x1, x2)

def torch_permute(x, dims):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.permute"](x, dims)
    return jnp.transpose(x, dims)

def torch_pow(x, exponent):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.pow"](x, exponent)
    return jnp.power(x, exponent)

def torch_select(x, dim, index):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.select"](x, dim, index)
    # TODO: likely inefficient. What's the better way?
    return lax.slice_in_dim(x, index, index+1, stride=1, axis=dim)[0]

def torch_slice(x, dim, start, end, step=1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.slice"](x, dim, start, end, step=step)
    if end > x.shape[dim]:
        end = x.shape[dim]
    return lax.slice_in_dim(x, start, end, stride=step, axis=dim)

def torch_split(x, split_size_or_sections, dim=0):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.split"](x, split_size_or_sections, dim=dim)
    if type(split_size_or_sections) == int:
        split_size = split_size_or_sections
        sections = list(range(split_size, x.shape[dim], split_size))
    else:
        assert type(split_size_or_sections) == list
        sections = split_size_or_sections
    return jnp.split(x, sections, axis=dim)

def torch_sqrt(x):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.sqrt"](x)
    return jnp.sqrt(x)

def torch_sub(x, other, alpha=1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.sub"](x, other, alpha=alpha)
    return x - alpha * other

def torch_sum(x, dim, keepdim=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.sum"](x, dim, keepdim=keepdim)
    return jnp.sum(x, axis=dim, keepdims=keepdim)

def torch_t(x):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.t"](x)
    return jnp.transpose(x)

def torch_transpose(x, dim0, dim1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.transpose"](x, dim0, dim1)
    return jnp.swapaxes(x, dim0, dim1)

def torch_unbind(x, dim=0):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.unbind"](x, dim=dim)
    ret = []
    for index in range(x.shape[dim]):
        # TODO: likely inefficient. What's the better way?
        ret.append(lax.slice_in_dim(x, index, index+1, stride=1, axis=dim)[0])
    return tuple(ret)

def torch_view(x, shape):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.view"](x, shape)
    return lax.reshape(x, infer_size(shape, x.size))

def torch_softmax(x, dim):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.softmax"](x, dim)
    x_max = jnp.max(x, axis=dim, keepdims=True)[0]
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=dim, keepdims=True)

def torch_nn_functional_softmax(x, dim):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.functional.softmax"](x, dim)
    return torch_softmax(x=x, dim=dim)

def torch_dropout(x, p=0.5, training=True, inplace=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.dropout"](x, p=p, training=training, inplace=inplace)
    assert not inplace, "Inplace dropout is not supported"
    if p == 0.:
      return x
    if training:
        # Copied from flax.linen.Dropout impl
        keep_prob = 1. - p
        # NOTE: pass None for rng, since Alpa ignores it anyway.
        mask = jax.random.bernoulli(None, p=keep_prob, shape=x.shape)
        return lax.select(mask, x, jnp.zeros_like(x))
    else:
        return x

def torch_nn_functional_dropout(x, p=0.5, training=True, inplace=False):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.functional.dropout"](x, p=p, training=training, inplace=inplace)
    return torch_dropout(x, p=p, training=training, inplace=inplace)

def torch_abs(x):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.abs"](x)
    return jnp.absolute(x)

def torch__normalize(x, mean, var, weight, bias, reduction_axes, feature_axes, eps):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch._normalize"](x, mean, var, weight, bias, reduction_axes, feature_axes, eps)
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
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.batch_norm"](
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )

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

def torch_relu(x):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.relu"](x)
    return jax.nn.relu(x)

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

def torch_flatten(x, start_dim=0, end_dim=-1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.flatten"](x, start_dim=start_dim, end_dim=end_dim)
    input_shape = x.shape
    start_dim = maybe_wrap_dim(start_dim, len(input_shape))
    end_dim = maybe_wrap_dim(end_dim, len(input_shape))
    assert start_dim <= end_dim
    if start_dim == end_dim:
        return x
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input_shape[i]
    shape = []
    for i in range(start_dim):
        shape.append(input_shape[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input_shape)):
        shape.append(input_shape[i])
    return torch_view(x, shape)

# TODO: dedup this with impl in decomposition
def torch_addmm(x, mat1, mat2, beta=1, alpha=1):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.addmm"](x, mat1, mat2, beta=beta, alpha=alpha)
    out = alpha * torch.matmul(mat1, mat2)
    if beta == 0:
        return out
    return beta * x + out

# TODO: dedup this with impl in decomposition
def torch_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.layer_norm"](x, normalized_shape, weight=weight, bias=bias, eps=eps, cudnn_enable=cudnn_enable)
    # TODO: this formula might be wrong
    axis = len(x.shape) - len(normalized_shape)
    mean_val = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.mean((x - mean_val) ** 2, axis=axis, keepdims=True)
    out = (x - mean_val) / jnp.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out

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
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.functional.batch_norm"](
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )
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
    x: torch.Tensor,
    target: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.functional.mse_loss"](
            x,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )
    # TODO: add handling for `size_average` / `reduce` / `reduction`
    return jnp.mean((x - target) ** 2)

# TODO: dedup this with decomposition.py
def torch_nn_functional_linear(x, weight, bias=None):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.functional.linear"](x, weight, bias=bias)
    output = torch.matmul(x, torch.t(weight))
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

def torch_nn_init_xavier_uniform(x, gain: float = 1.):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.init.xavier_uniform"](x, gain=gain)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(x)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return DistributedArray.init_like(
        x,
        jax.random.uniform,
        key=None,  # TODO: is random key already handled by Alpa?
        minval=-a,
        maxval=a,
    )

def torch_nn_init_normal(x, mean: float = 0., std: float = 1.):
    if is_torch_tensor_type(x):
        return op_orig_impl_dict["torch.nn.init.normal"](x, mean=mean, std=std)
    return DistributedArray.init_like(
        x,
        jax.random.normal,
        key=None,  # TODO: is random key already handled by Alpa?
        mu=mean,
        sigma=std,
    )

def alpa_automatic_layer_construction(fun=None, **kwargs):
    if atorch.mode() == "local":
        def decorate_fun(fun):
            return fun

        if fun is None:
            return decorate_fun
        else:
            return decorate_fun(fun)
    return op_orig_impl_dict["alpa.automatic_layer_construction"](fun=fun, **kwargs)


# PyTorch .detach() is equivalent to JAX lax.stop_gradient(): https://github.com/google/jax/issues/2025
# PyTorch .view() is equivalent to JAX lax.reshape(): https://jax.readthedocs.io/en/latest/_autosummary/lax.reshape.html


op_orig_impl_dict = {}
op_patch_list = [
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
    (torch, "layer_norm", torch_layer_norm),
    (torch.nn.functional, "softmax", torch_nn_functional_softmax),
    (torch.nn.functional, "dropout", torch_nn_functional_dropout),
    (torch.nn.functional, "batch_norm", torch_nn_functional_batch_norm),
    (torch.nn.functional, "mse_loss", torch_nn_functional_mse_loss),
    (torch.nn.functional, "linear", torch_nn_functional_linear),
    (torch.nn.init, "xavier_uniform", torch_nn_init_xavier_uniform),
    (torch.nn.init, "normal", torch_nn_init_normal),
    # (torch.nn.init, "zeros", torch_nn_init_zeros),
    (alpa, "automatic_layer_construction", alpa_automatic_layer_construction),
    # TODO: add hard error for in-place ops
]


def patch_ops():
    for python_module, op_name, new_impl in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        op_orig_impl_dict[f"{python_module_fqn}.{op_name}"] = getattr(python_module, op_name, None)
        setattr(python_module, op_name, new_impl)


def unpatch_ops():
    for python_module, op_name, _ in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        op_orig_impl = op_orig_impl_dict.get(f"{python_module_fqn}.{op_name}", None)
        if op_orig_impl is not None:
            setattr(python_module, op_name, op_orig_impl)
        else:
            delattr(python_module, op_name)


@contextlib.contextmanager
def bind_ops():
    """
    Context manager within which many PyTorch ops are monkey-patched to also accept Alpa arrays.

    Also disable the `alpa.automatic_layer_construction()` API in local mode.
    """
    patch_ops()
    yield
    unpatch_ops()
