"""Test auto sharding with convoluton nets."""
from typing import Any

from flax import linen as nn, optim
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from alpa import parallelize, global_config
from alpa.util import count_communication_primitives

batch_size = 32
image_size = 32
channel = 512
use_bias = True


class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        last_x = None
        for i in range(10):
            x = nn.Conv(features=channel,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        use_bias=use_bias)(x)
            x += jnp.sum(x)
            #x = nn.relu(x)
            #if last_x is not None:
            #    x = x + last_x
            #last_x = x
        return x

@parallelize
def train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {
    "x": jnp.ones((batch_size, image_size, image_size, channel)),
    "y": jnp.ones((batch_size, image_size, image_size, channel))
}


# Init train state
model = Model()
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, batch["x"])
tx = optax.sgd(0.1)
state = TrainState.create(apply_fn=model.apply,
                          params=params,
                          tx=tx)

executable = train_step.get_executable(state, batch)

hlo_ir = executable.get_hlo_text()
n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ = (
    count_communication_primitives(hlo_ir))
print(n_total, n_all_reduce)

# JIT compile
ct = 0
while True:
    state = train_step(state, batch)
    if ct % 100 == 0:
        print(ct)
    ct += 1
