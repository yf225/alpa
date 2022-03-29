"""Test auto sharding with convoluton nets."""
from functools import partial

from flax import jax_utils
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

batch_size = 32
hidden_size = 32
use_bias = True


class Model(nn.Module):

    @nn.compact
    def __call__(self, x, call_sum=False):
        for i in range(10):
            x = nn.Dense(hidden_size, use_bias=use_bias)(x)
            if call_sum:
                x += jax.lax.psum(x, "b")
        return x


@partial(jax.pmap, axis_name="b")
def train_step(state, batch):
    out = state.apply_fn(params, batch["x"], call_sum=True)
    loss = jnp.mean((out - batch["y"])**2)
    return loss


num_devices = len(jax.local_devices())
batch = {
    "x": jnp.ones((num_devices, batch_size, hidden_size)),
    "y": jnp.ones((num_devices, batch_size, hidden_size))
}

# Init train state
model = Model()
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, batch["x"])
tx = optax.sgd(0.1)
state = TrainState.create(apply_fn=model.apply,
                          params=params,
                          tx=tx)
state = jax_utils.replicate(state)

# Run
ct = 0
while True:
    train_step(state, batch)
    if ct % 500 == 0:
        print(ct)
    ct += 1
