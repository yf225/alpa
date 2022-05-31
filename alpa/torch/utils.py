import torch
import numpy as np
import jax


# Copied from https://github.com/pytorch/pytorch/blob/ac79c874cefee2f8bc1605eed9a924d80c0b3542/torch/testing/_internal/common_utils.py#L349
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def make_shaped_array_from_pt_tensor(pt_tensor):
    shape = list(pt_tensor.shape)
    np_dtype = torch_to_numpy_dtype_dict[pt_tensor.dtype]
    return jax.abstract_arrays.ShapedArray(shape, np_dtype)
