import torch
import alpa
import alpa.torch as atorch
from torch_trainer import train_torch_module


B = 64

# Define a very simple PyTorch model
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.reshape(x.shape[0], 2, -1)
        x = x.reshape(x.shape[0], -1, 2)
        x = x.reshape(x.shape[0], 16)
        return x


# `meta_init` allows a PyTorch model to be created with shape-only tensors as weights.
pt_module = alpa.torch.meta_init(MyModule)


# Define the weight initialization function
def weight_init_func(pt_module, name_map, params, bufs):
    # First, materialize all weights to zero
    for k, p in pt_module.named_parameters():
        params[name_map[f"{k}"]] = atorch.zeros_like(params[name_map[f"{k}"]])
    for k, b in pt_module.named_buffers():
        bufs[name_map[f"{k}"]] = atorch.zeros_like(bufs[name_map[f"{k}"]])
    return params, bufs


dataloader = [
    (torch.randn(B, 16), torch.randn(B, 16)),
    (torch.randn(B, 16), torch.randn(B, 16)),
]


train_torch_module(pt_module, weight_init_func, dataloader)
