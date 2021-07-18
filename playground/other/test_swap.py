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

from parax.model.bert_model import BertConfig, FlaxBertLayer
from parax.xla_pass_context import XlaPassContext

from timeit import timeit

def test_bert_layer():
    batch_size = 64
    seq_len = 64
    hidden_size = 768

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    # Init model and optimizer
    model = FlaxBertLayer(BertConfig(
        hidden_size=hidden_size, attention_probs_dropout_prob=0))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
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
      "swap::bound": 170 * 1024 * 1024  
      # currently, the bound ignores input parameter size, 
      # so there is a gap between the bound and total allocation size
    }):
      compiled_computation = gpu_backend.compile(c)
    print(compiled_computation.total_allocation_size())
    # print(compiled_computation.hlo_modules()[0].to_string())

    host_inputs = [np.ones((),dtype=np.int32),
    np.ones((768),dtype=np.float32),
    np.ones((768),dtype=np.float32),
    np.ones((768),dtype=np.float32),
    np.ones((768, 768),dtype=np.float32),
    np.ones((2304),dtype=np.float32),
    np.ones((768, 2304),dtype=np.float32),
    np.ones((3072),dtype=np.float32),
    np.ones((768, 3072),dtype=np.float32),
    np.ones((768),dtype=np.float32),
    np.ones((768),dtype=np.float32),
    np.ones((768),dtype=np.float32),
    np.ones((3072, 768),dtype=np.float32),
    np.ones((64, 64),dtype=np.int32),
    np.ones((64, 64, 768),dtype=np.float32),
    np.ones((64, 64, 768),dtype=np.float32),
    np.ones((2),dtype=np.uint32)]
    device_input = [gpu_backend.buffer_from_pyval(x) for x in host_inputs]
    compiled_computation.execute(device_input)
    compiled_computation.execute(device_input)
    
    
    stmt = "compiled_computation.execute(device_input)"
    number = 100
    time_cost = timeit(stmt, globals={**globals(), **locals()},
                       number=number) / number
    print(time_cost)

    """----------------results:----------------"""

    # testbench: 2070 SUPER(f32 9.1 TFLOPS, PCIe3.0*16 with 16GB/s)
    # without swap: 
    # 0.027576977228745816s, 346143384 Bytes, set to 400

    # with swap:
    # 0.026839641025289893s, 308394648 Bytes, set to 320

    # with swap:
    # 0.027199601493775843s, 320977560 Bytes, set to 300

    # with swap: 
    # 0.02660452459938824s, 295811736 Bytes, set to: 260

    # with swap: 
    # 0.026400231635197998s, 270645912 Bytes, set to: 200

    # with swap: 
    # 0.02796688610687852s, 245480088 Bytes, set to: 170
    # 1.3% more time(5.8%compared with swap setting 200), 29% less memory

    # with swap: 
    # 0.04833599615842104s, 245463704 Bytes, set to: 150
    # warning: memory bound is impossible. 

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    jax.config.update('jax_platform_name', 'cpu')
    # ray.init(address="auto")
    test_bert_layer()