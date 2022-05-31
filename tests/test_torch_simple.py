import torch
import alpa
import alpa.torch as atorch
from torch_trainer import train_torch_module

# Define a very simple PyTorch model
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 17)
        self.linear2 = torch.nn.Linear(17, 18)
    def forward(self, x):
        x = self.linear1(x)
        # do some debugging when in local mode
        if alpa.torch.mode() == "local":
            print(x)
        x = self.linear2(x)
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

    # Then, selectively initialize some weights to a different value
    for k, m in pt_module.named_modules():
        if isinstance(m, torch.nn.Linear):
            params[name_map[f"{k}.weight"]] = torch.nn.init.xavier_uniform(params[name_map[f"{k}.weight"]])
            params[name_map[f"{k}.bias"]] = torch.nn.init.normal(params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


dataloader = [
    (torch.randn(8, 16), torch.randn(8, 18)),
    (torch.randn(8, 16), torch.randn(8, 18)),
]

train_torch_module(pt_module, weight_init_func, dataloader)
