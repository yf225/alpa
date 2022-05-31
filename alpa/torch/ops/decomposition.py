import torch
from typing import List, Dict, Any
import contextlib
from alpa.torch.ops.utils import infer_size


@torch.fx.wrap
def torch_mean(input: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    return torch.mean(input, dim, keepdim=keepdim)


def torch_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True):
    # TODO: this formula might be wrong
    axis = x.dim() - len(normalized_shape)
    mean_val = torch_mean(x, dim=axis, keepdim=True)
    var = torch.square(x - mean_val).mean(dim=axis, keepdim=True)
    out = (x - mean_val) / torch.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def torch_nn_functional_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True):
    return torch_layer_norm(
        x, normalized_shape, weight=weight, bias=bias, eps=eps, cudnn_enable=cudnn_enable
    )


def torch_nn_functional_linear(input, weight, bias=None):
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output


def torch_square(input):
    return torch.pow(input, 2)


def torch_addmm(input, mat1, mat2, beta=1, alpha=1):
    out = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return out
    return beta * input + out


# def torch_Tensor_view(tensor, *shape):
#     # Infer full shape based on input tensor, then set batch dim size to -1
#     shape = list(shape)
#     if -1 in shape:
#         shape = infer_size(shape, tensor.numel())
#     shape[0] = -1
#     return op_orig_impl_dict["torch.Tensor.view"](tensor, *shape)


# def torch_Tensor_reshape(tensor, *shape):
#     # Infer full shape based on input tensor, then set batch dim size to -1
#     shape = list(shape)
#     if -1 in shape:
#         shape = infer_size(shape, tensor.numel())
#     shape[0] = -1
#     return op_orig_impl_dict["torch.Tensor.reshape"](tensor, *shape)


# def addmm(input, mat1, mat2, *, beta=1, alpha=1):
#     return beta * input + alpha * torch.mm(mat1, mat2)


# def softmax(x, dim):
#     x_max = torch.max(x, dim, keepdim=True)[0]
#     unnormalized = torch.exp(x - x_max)
#     return unnormalized / torch.sum(unnormalized, dim, keepdim=True)


# def log_softmax(x, dim):
#     x_max = torch.max(x, dim, keepdim=True)[0]
#     shifted = x - x_max
#     shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
#     return shifted - shifted_logsumexp


# def addcdiv(self, tensor1, tensor2, value=1):
#     return self + value * (tensor1 / tensor2)


# def addcmul(self, tensor1, tensor2, value=1):
#     if self.is_floating_point():
#         return self + value * tensor1 * tensor2
#     else:
#         return self + int(value) * tensor1 * tensor2


op_orig_impl_dict = {}
op_patch_list = [
    # Decompose ops
    (torch.nn.functional, "layer_norm", torch_nn_functional_layer_norm),
    (torch, "layer_norm", torch_layer_norm),
    (torch.nn.functional, "linear", torch_nn_functional_linear),
    (torch, "square", torch_square),
    (torch, "addmm", torch_addmm),
    # (torch.Tensor, "view", torch_Tensor_view),
    # (torch.Tensor, "reshape", torch_Tensor_reshape),
    # (torch.nn.functional, "softmax", softmax),  # why decomp like this doesn't work?
    # (torch.nn.functional, "log_softmax", log_softmax),
    # (torch, "addcdiv", addcdiv),
    # (torch, "addcmul", addcmul),
]


def get_python_module_fqn(python_module):
    if "<module " in str(python_module):
        python_module_fqn = str(python_module).split("<module '")[1].split("'")[0]
    elif "<class " in str(python_module):
        python_module_fqn = str(python_module).split("<class '")[1].split("'")[0]
    return python_module_fqn


def patch_ops():
    for python_module, op_name, new_impl in op_patch_list:
        python_module_fqn = get_python_module_fqn(python_module)
        op_orig_impl_dict[f"{python_module_fqn}.{op_name}"] = getattr(python_module, op_name)
        setattr(python_module, op_name, new_impl)


def unpatch_ops():
    for python_module, op_name, _ in op_patch_list:
        python_module_fqn = get_python_module_fqn(python_module)
        setattr(python_module, op_name, op_orig_impl_dict[f"{python_module_fqn}.{op_name}"])


@contextlib.contextmanager
def decompose_ops():
    # Decompose big ops into multiple smaller ops.
    #
    # Could be a good way to reduce the number of PyTorch->JAX operator mappings needed,
    # or as a way to add Meta support without needing to add shape-inference-only operator impl.
    patch_ops()
    yield
    unpatch_ops()
