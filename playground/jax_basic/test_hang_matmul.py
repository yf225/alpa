"""Test auto sharding with convoluton nets."""
from functools import partial

import jax
import jax.numpy as jnp

batch_size = 32
hidden_size = 128

@partial(jax.pmap, axis_name="b")
def train_step(params, x):
    out = x
    for i in range(len(params)):
        out = out @ params[i]
        out += jax.lax.psum(x, "b")
    return out

num_devices = len(jax.local_devices())

x = jnp.ones((num_devices, batch_size, hidden_size))
params = []
for i in range(10):
    params.append(jnp.ones((num_devices, hidden_size, hidden_size)))

# Run
ct = 0
while True:
    train_step(params, x)
    if ct % 500 == 0:
        print(ct)
    ct += 1
