import os

from jax._src.numpy.lax_numpy import block
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
from functools import partial

import numpy as np

from flax import optim
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.lib import xla_client
from jaxlib.xla_extension import DeviceArray

from parax.model.bert_model import BertConfig, FlaxBertLayerCollection
from parax.xla_pass_context import XlaPassContext

from timeit import timeit

MB = 1024 ** 2
no_remat_mem = None
no_remat_time = None
def test_bert_layer(enable_checkpoint = False):
    global no_remat_mem
    global no_remat_time
    batch_size = 32
    seq_len = 256
    hidden_size = 768
    num_hidden_layers = 8
    num_heads = 768 // 96

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    # Init model and optimizer
    # model = FlaxBertLayer(BertConfig(
    #     hidden_size=hidden_size, attention_probs_dropout_prob=0))
    model = FlaxBertLayerCollection(
      BertConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        enable_checkpoint=enable_checkpoint
        )
    )
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
    flatten_params = jax.tree_flatten(params.tree_flatten()[0][0]['params'], lambda x : x is DeviceArray)[0]

    optimizer = optim.GradientDescent(1e-2).create(params)
    def train_step(optimizer, batch):
        def loss_func(params):
            rngs = {"dropout": batch['rng']}
            out = model.apply(params, batch['hidden_states'],
                              batch['attention_mask'],
                              rngs=rngs)[0]
            return jnp.mean((out - batch['label']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    c = jax.xla_computation(train_step)(optimizer,
                            {"hidden_states": hidden_states,
                              "attention_mask": attention_mask,
                              "label": label,
                              "rng": rngkey})

    gpu_backend = xla_client.get_local_backend("gpu")
    compiled_computation = gpu_backend.compile(c)
    # print(compiled_computation.hlo_modules()[0].to_string())

    host_inputs = [np.ones((),dtype=np.int32)] + flatten_params +\
    [attention_mask, hidden_states, label, np.ones((2), dtype=np.uint32)]
    device_input = [gpu_backend.buffer_from_pyval(x) for x in host_inputs]

    size = compiled_computation.total_allocation_size()
    
    compiled_computation.execute(device_input)
    compiled_computation.execute(device_input)
    stmt = "compiled_computation.execute(device_input)"
    number = 30
    time_cost = timeit(stmt, globals={**globals(), **locals()},
                       number=number) / number
    print("remat: " if enable_checkpoint else "no remat: ", size / MB, "MB,", time_cost, "seconds")
    if enable_checkpoint:
      if no_remat_mem and no_remat_time:
        print(size / no_remat_mem * 100, "percents of space,", no_remat_time / time_cost * 100, "percents of speed")
    else:
      no_remat_mem = size
      no_remat_time = time_cost

if __name__ == "__main__":
    test_bert_layer(False)
    test_bert_layer(True)