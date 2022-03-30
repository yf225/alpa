from functools import partial

import jax
import jax.numpy as jnp

batch_size = 32
hidden_size = 128

@partial(jax.pmap, axis_name="b")
def train_step(params, x):
    out = x
    for i in range(10):
        out = out @ params
        out += jax.lax.psum(out, "b")
    return out

num_devices = len(jax.local_devices())

x = jnp.ones((num_devices, batch_size, hidden_size))
params = jnp.ones((num_devices, hidden_size, hidden_size))

# Run
ct = 0
while True:
    train_step(params, x)
    if ct % 500 == 0:
        print(ct)
    ct += 1
