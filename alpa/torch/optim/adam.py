import copy

import jax
import torch
import alpa.torch as atorch


def adam(params, lr=1e-4):
    """
    alpa.torch.optim.adam(params, **adam_config)
        Factory that generates Alpa-compatible Adam optimizer that accepts both PT and Alpa inputs.
        If `alpa.torch.mode` is "local", assumes input is in PT format. Otherwise assumes input is Alpa format.
        Implementation has no in-place op and no data-dependent control flow.
        NOTE: we will need similar implementation for other torch.optim optimizers.
        Returns:
            - `optim_func`: a function that:
                - takes (`params`, `params_grad`, `optim_state`) as input
                - returns (`params`, `optim_state`) as result of applying Adam algorithm
            - `optim_state`: tracked state (shape-only tensors) of Adam optimizer.
                If `alpa.torch.mode` is "local", `optim_state` is in PT format.
                Otherwise if `alpa.torch.mode` is "dist", `optim_state` is in Alpa format.
            - `optim_state_init_func`: a function that:
                - takes `optim_state` as input (PT format in "local" mode, Alpa format in "dist" mode)
                - returns `optim_state` as result of materializing the Adam optimizer state
    """
    # TODO FIXME: properly implement Adam optimizer
    def optim_func(optim_state, params, params_grad):
        for k in params:
            params[k] = params[k] + optim_state[k]
            optim_state[k] = optim_state[k] + 1
        return params, optim_state

    optim_state = copy.deepcopy(params)

    def optim_state_init_func(optim_state):
        new_state = {}
        for k, v in optim_state.items():
            if atorch.mode() == "local":
                new_state[k] = torch.full_like(v, 0.0, device="cpu")
            elif atorch.mode() == "dist":
                # try:
                #     new_state[k] = jax.numpy.full_like(v, 0.0)
                # except:  # TODO: check obj type instead
                #     new_state[k] = jax.numpy.full(v.shape, 0.0, dtype=v.dtype)
                new_state[k] = jax.numpy.full_like(v, 0.0)
        return new_state

    return optim_func, optim_state, optim_state_init_func
