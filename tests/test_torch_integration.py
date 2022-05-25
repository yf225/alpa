import unittest
import functools
import os

import torch
assert "dev" in torch.__version__, """
Only PyTorch nightly version is supported for now.

Install PyTorch nightly version with:

pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu

or see https://pytorch.org/.
"""

import alpa
import ray


try:
    import torchdynamo
except ImportError as e:
    print("""
Please install torchdynamo:

cd ${HOME_DIR}
git clone https://github.com/facebookresearch/torchdynamo
cd ${HOME_DIR}/torchdynamo
pip install -U setuptools
pip install -r requirements.txt
python3 setup.py install

""")
    raise e

try:
    import functorch
except ImportError as e:
    print("""
Please install functorch:

cd ${HOME_DIR}
git clone https://github.com/pytorch/functorch
cd ${HOME_DIR}/functorch
python3 setup.py install

""")
    raise e


try:
    from torchdistx import deferred_init
except ImportError as e:
    print("""
Please install torchdistx:

cd ${HOME_DIR}
git clone https://github.com/pytorch/torchdistx
cd ${HOME_DIR}/torchdistx
git submodule update --init --recursive
cmake -DTORCHDIST_INSTALL_STANDALONE=ON -B build
cmake --build build
pip install -U setuptools
pip install .

""")
    raise e


class TorchIntegrationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        import jax
        assert len(jax.local_devices()) >= 4

        ray.init(address="auto",
                 namespace=alpa.util.get_ray_namespace_str(prefix="alpa-unittest"))

    def tearDown(self):
        ray.shutdown()

    def run_train_flow(self, mode):
        # "local": pure PT eager mode on single GPU, allows print in middle of graph, no dist training
        # "dist": graph mode by lowering PT program to JAX, doesn't allow print, dist training
        # NOTE: as we see below, the two modes can share most of the code.
        alpa.torch.set_mode(mode)


        if alpa.torch.mode() == "dist":
            # Set up necessary Alpa config for dist training.
            physical_mesh = alpa.DeviceCluster().get_virtual_physical_mesh()
            alpa.set_parallelize_options(
                devices=physical_mesh,
                strategy="pipeshard_parallel",
                pipeline_stage_mode="auto_stage",
                num_micro_batches=2,
            )


        # This sets both `torch.manual_seed` and `alpa.api.manual_seed` under the hood.
        alpa.torch.manual_seed(seed=123)


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
            # `alpa.torch.bind_ops` enables torch operators to be called with Alpa tensors
            with alpa.torch.bind_ops():
                # First, materialize all weights to zero
                for k, p in pt_module.named_parameters():
                    params[name_map[f"{k}"]] = torch.nn.init.zeros(params[name_map[f"{k}"]])
                for k, b in pt_module.named_buffers():
                    bufs[name_map[f"{k}"]] = torch.nn.init.zeros(bufs[name_map[f"{k}"]])

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


        # Assume we have a dataloader that supports `peek` function
        # (i.e. look at next batch but don't advance the pointer)
        pt_inputs, pt_targets = dataloader[0]  # dataloader.peek()

        # Create shape-only version of input
        pt_inputs = torch.empty_like(pt_inputs, device="meta")
        pt_targets = torch.empty_like(pt_targets, device="meta")

        # Functionalize the PyTorch model
        module_func, params, bufs, name_map = alpa.torch.nn.make_func(pt_module, pt_inputs)

        # Use functional version of loss function
        loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(*args, **kwargs)

        # Use functional version of optimizer
        # `torch.optim.*` optimizers have in-place ops, so have to implement our own version of optimizer.
        optim_func, optim_state, optim_state_init_func = alpa.torch.optim.adam(params, lr=1e-4)


        # Define the training loop
        def sgd_train_func(module_func, loss_func, optim_func, params, bufs, optim_state, batch):
            inputs = batch[0]
            targets = batch[1]

            # wrap forward pass + loss computation in a function
            def compute_loss(params, bufs, inputs, targets):
                # do forward pass
                bufs, out = module_func(params, bufs, inputs)

                # do some debugging when in local mode
                if alpa.torch.mode() == "local":
                    print("out: ", out)

                # do loss computation
                loss_value = loss_func(out, targets)
                return loss_value, bufs

            if alpa.torch.mode() == "dist":
                compute_loss = alpa.automatic_layer_construction(layer_num="auto")(compute_loss)

            # do model forward + backward pass
            params_grad, (loss_value, bufs) = functorch.grad_and_value(compute_loss, has_aux=True)(params, bufs, inputs, targets)

            # do optimizer step
            params, optim_state = optim_func(params, params_grad, optim_state)

            return params, bufs, optim_state, loss_value


        train_func = functools.partial(sgd_train_func, module_func, loss_func, optim_func)

        def iter_func(params, bufs, optim_state, pt_batch):
            # `alpa.torch.bind_ops` enables torch operators to be called with Alpa tensors
            with alpa.torch.bind_ops():
                if alpa.torch.mode() == "local":  # local development
                    params, bufs, optim_state, loss_value = train_func(
                        params,
                        bufs,
                        optim_state,
                        pt_batch,
                    )
                    # Show that outputs are in PT format (no need in actual training code)
                    alpa.torch.assert_format("torch", params, bufs, optim_state, loss_value)
                    pt_loss_value = loss_value
                elif alpa.torch.mode() == "dist":  # distributed training
                    params, bufs, optim_state, loss_value = alpa.api.parallelize(
                        train_func,
                        batch_argnums=(3,),  # NOTE: assumes the 4th argument is input batch
                        donate_argnums=(0, 1, 2),  # NOTE: preserves memory addr and sharding spec for first 3 args
                    )(
                        params,
                        bufs,
                        optim_state,
                        alpa.torch.to_format("alpa", pt_batch),
                    )
                    # Show that outputs are in Alpa format (no need in actual training code)
                    alpa.torch.assert_format("alpa", params, bufs, optim_state, loss_value)
                    pt_loss_value = alpa.torch.to_format("torch", loss_value)
                return params, bufs, optim_state, pt_loss_value


        # Generate sharding plan based on shape-only tensors. Only needed for dist training.
        if alpa.torch.mode() == "dist":
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = True
            params, bufs, optim_state, _ = iter_func(params, bufs, optim_state, (pt_inputs, pt_targets))
            alpa.global_env.global_config.use_dummy_value_for_benchmarking = False

        # Materialize the weights and optimizer state
        params, bufs = weight_init_func(pt_module, name_map, params, bufs)
        optim_state = optim_state_init_func(optim_state)

        # Run training loops
        for i, (pt_inputs, pt_targets) in enumerate(dataloader):
            params, bufs, optim_state, pt_loss_value = iter_func(
                params, bufs, optim_state, (pt_inputs, pt_targets)
            )
            # do whatever with the loss value, e.g. plot it on a graph
            print("loss value: ", pt_loss_value)


    def test_local_mode(self):
        self.run_train_flow("local")


    def test_dist_mode(self):
        self.run_train_flow("dist")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TorchIntegrationTest("test_local_mode"))
    suite.addTest(TorchIntegrationTest("test_dist_mode"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
