import torch
from typing import List, Dict, Any
import contextlib


# All patched ops in this file check this global variable to decide whether to use
# shape-inference-only op impl or original op impl
shape_inference_only = {"enabled": False}


op_orig_impl_dict = {}


# Copied from pytorch/aten/src/ATen/native/ConvUtils.h
input_batch_size_dim = 0  # also grad_input
input_channels_dim = 1
output_batch_size_dim = 0  # also grad_output
output_channels_dim = 1
weight_output_channels_dim = 0
weight_input_channels_dim = 1


# Copied from pytorch/aten/src/ATen/native/ConvUtils.h
def _conv_output_size(
    input_size: List[int], weight_size: List[int],
    padding: List[int], stride: List[int], dilation: List[int] = []
) -> List[int]:
    # assert len(input_size) > 2
    # assert len(input_size) == len(weight_size)
    has_dilation: bool = len(dilation) > 0;
    dim = len(input_size)
    output_size = [0] * dim
    output_size[0] = input_size[input_batch_size_dim]
    output_size[1] = weight_size[weight_output_channels_dim]
    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size[d] = int((input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1)
    return output_size


# Copied from pytorch/aten/src/ATen/native/ConvUtils.h
def _conv_input_size(
    output_size: List[int], weight_size: List[int],
    padding: List[int], output_padding: List[int], stride: List[int], dilation: List[int], groups: int
) -> List[int]:
    # assert len(output_size) > 2
    # assert len(output_size) == len(weight_size)
    dim = len(output_size)
    input_size = [0] * dim
    input_size[0] = output_size[output_batch_size_dim]
    input_size[1] = weight_size[weight_input_channels_dim] * groups
    for d in range(2, dim):
        kernel = dilation[d - 2] * (weight_size[d] - 1) + 1
        input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) + \
                         kernel + output_padding[d - 2]
    return input_size


# Copied from pytorch/aten/src/ATen/native/Convolution.cpp
def _calc_output_size(
    input: torch.Tensor,
    weight: torch.Tensor,
    params: Dict[str, Any]) -> List[int]:
    output_size: List[int] = _conv_input_size(
        input.shape, weight.shape, params["padding"], params["output_padding"],
        params["stride"], params["dilation"], params["groups"]
    ) if params["transposed"] else _conv_output_size(
        input.shape, weight.shape, params["padding"], params["stride"], params["dilation"]
    )

    # Handle empty # of channels.
    if input.shape[1] == 0:
        output_size[input_channels_dim] = 0
    return output_size


# Copied from pytorch/aten/src/ATen/native/utils/ParamUtils.h
def _expand_param_if_needed(
    list_param: List[int],
    param_name: str,
    expected_dim: int,
) -> List[int]:
    if len(list_param) == 1:
        return [list_param[0]] * expected_dim
    elif len(list_param) != expected_dim:
        raise ValueError(
            f"expected {param_name} to be a single integer value or a " + \
            "list of {expected_dim} values to match the convolution " + \
            "dimensions, but got {param_name} = {list_param}"
        )
    else:
        return list_param


# Copied from pytorch/aten/src/ATen/native/Convolution.cpp
def torch_nn_functional_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if shape_inference_only["enabled"]:
        k = weight.ndim
        dim: int = k - 2
        out_shape = _calc_output_size(input, weight, dict(
            stride=_expand_param_if_needed(stride, "stride", dim),
            padding=_expand_param_if_needed(padding, "padding", dim),
            dilation=_expand_param_if_needed(dilation, "dilation", dim),
            transposed=False,
            output_padding=[],
            groups=groups,
        ))
        return torch.empty(out_shape, device=torch.device("meta"), dtype=input.dtype)
    else:
        return op_orig_impl_dict["torch.nn.functional.conv2d"](
            input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=dilation
        )


def torch_cat(tensors, dim=0):
    if shape_inference_only["enabled"]:
        for t in tensors:
            if t.numel() > 0:
                out_shape = list(t.shape)
                break
        out_shape[dim] = 0
        for t in tensors:
            if t.numel() > 0:
                out_shape[dim] += t.shape[dim]
        return torch.empty(out_shape, device=torch.device("meta"), dtype=tensors[0].dtype)
    else:
        return op_orig_impl_dict["torch.cat"](tensors, dim=dim)


def torch_abs(input):
    if shape_inference_only["enabled"]:
        return torch.empty(input.shape, device=torch.device("meta"), dtype=input.dtype)
    else:
        return op_orig_impl_dict["torch.abs"](input)


def torch_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    if shape_inference_only["enabled"]:
        return torch.empty(input.shape, device=torch.device("meta"), dtype=input.dtype)
    else:
        return op_orig_impl_dict["torch.batch_norm"](
            input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled
        )

def torch_relu(input, inplace=False):
    if shape_inference_only["enabled"]:
        assert inplace is False
        return torch.empty(input.shape, device=torch.device("meta"), dtype=input.dtype)
    else:
        return op_orig_impl_dict["torch.relu"](
            input, inplace=inplace
        )


op_patch_list = [
    # Shape-inference only
    (torch.nn.functional, "conv2d", torch_nn_functional_conv2d),
    (torch, "cat", torch_cat),
    (torch, "abs", torch_abs),
    (torch, "batch_norm", torch_batch_norm),
    (torch, "relu", torch_relu),
]


def patch_ops():
    for python_module, op_name, new_impl in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        op_orig_impl_dict[f"{python_module_fqn}.{op_name}"] = getattr(python_module, op_name)
        setattr(python_module, op_name, new_impl)


def unpatch_ops():
    for python_module, op_name, _ in op_patch_list:
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
        setattr(python_module, op_name, op_orig_impl_dict[f"{python_module_fqn}.{op_name}"])


@contextlib.contextmanager
def infer_output_shape():
    # Use shape-inference-only impl for ops such as `conv2d`.
    global shape_inference_only
    patch_ops()
    shape_inference_only["enabled"] = True
    yield
    shape_inference_only["enabled"] = False
    unpatch_ops()
