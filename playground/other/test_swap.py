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

def test_bert_layer():
    batch_size = 64
    seq_len = 64
    hidden_size = 768
    num_hidden_layers = 3
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
        num_attention_heads=num_heads)
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
    with XlaPassContext({
      "swap::enable": True,
      "swap::bound": 400 * 1024 * 1024  
      # currently, the bound ignores input parameter size, 
      # so there is a gap between the bound and total allocation size
    }):
      compiled_computation = gpu_backend.compile(c)
    # print(compiled_computation.total_allocation_size())
    size = compiled_computation.total_allocation_size()
    # print(compiled_computation.hlo_modules()[0].to_string())

    host_inputs = [np.ones((),dtype=np.int32)] + flatten_params +\
    [attention_mask, hidden_states, label, np.ones((2), dtype=np.uint32)]
    device_input = [gpu_backend.buffer_from_pyval(x) for x in host_inputs]
    compiled_computation.execute(device_input)
    compiled_computation.execute(device_input)
    
    
    stmt = "compiled_computation.execute(device_input)"
    number = 100
    time_cost = timeit(stmt, globals={**globals(), **locals()},
                       number=number) / number
    print(size, "Bytes,", time_cost, "seconds")
    print(size / 845623444 * 100, "percents of space,", 0.08257338968105614 / time_cost * 100, "percents of speed")

    """----------------results:----------------"""

    # testbench: 2070 SUPER(f32 9.1 TFLOPS, PCIe3.0*16 with 16GB/s)
    # 3 layers
    # without swap
    # 845623444 Bytes, 0.08257338968105614s

    # with swap(300MB)
    # 472248468 Bytes, 0.11426283085718751s
    # 55.85 percents of space, 72.31 percents of speed

    # with swap(400MB)
    # 585461908 Bytes, 0.09247550043277443s
    # 69.23  percents of space,  89.29  percents of speed

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    jax.config.update('jax_platform_name', 'cpu')
    # ray.init(address="auto")
    test_bert_layer()