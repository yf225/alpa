import torch


# Adapted from aten/src/ATen/InferSize.h infer_size_impl()
def infer_size(shape, numel):
    newsize = 1
    infer_dim = None
    len(shape)
    res = list(shape)
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise ValueError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise Exception(f"invalid shape dimension {shape[dim]}")

    if (numel == newsize) or (infer_dim is not None and newsize > 0 and numel % newsize == 0):
        if infer_dim is not None:
            # We have a degree of freedom here to select the dimension size; follow
            # NumPy semantics and just bail.  However, a nice error message is needed
            # because users often use `view` as a way to flatten & unflatten
            # dimensions and will otherwise be confused why
            #   empty_tensor.view( 0, 0)
            # works yet
            #   empty_tensor.view(-1, 0)
            # doesn't.
            assert newsize != 0, (
                "cannot reshape tensor of 0 elements into shape "
                + str(shape)
                + " because the unspecified dimension size -1 can be any "
                + "value and is ambiguous"
            )
            res[infer_dim] = numel // newsize
        return res

    raise Exception(f"shape {shape} is invalid for input of size {numel}")


def is_torch_tensor_type(x):
    return isinstance(x, (torch.Tensor, torch.fx.proxy.Proxy))
