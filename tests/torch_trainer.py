import functools

import torch
import alpa
import alpa.torch as atorch


"""
====== New APIs (added to `alpa.torch.*` namespace) ======
1. atorch.bind_ops()
2. atorch.set_mode()
3. atorch.mode()
4. atorch.to_format()
5. atorch.assert_format()
6. atorch.meta_init()
7. atorch.functionalize()
8. atorch.optim.adam()
9. atorch.grad_and_value()
10. atorch.manual_seed()
====== New APIs (added to `alpa.torch.*` namespace) ======
"""


def train_torch_module(pt_module, weight_init_func, dataloader):
    # for mode in ["local", "dist"]:
    for mode in ["dist"]:  # TODO: re-enable local mode when https://github.com/pytorch/torchdistx/issues/21 is fixed.
        # `atorch.bind_ops()` is no-op in local mode, and enables `torch.*` operators to be called with Alpa tensors in dist mode.
        with atorch.bind_ops():
            # "local": pure PT eager mode on single GPU, allows print in middle of graph, no dist training
            # "dist": graph mode by lowering PT program to JAX, doesn't allow print, dist training
            # NOTE: as we see below, the two modes can share most of the code.
            atorch.set_mode(mode)

            # Prints verbose log for debugging.
            atorch.debug = True

            if atorch.mode() == "dist":
                alpa.init(cluster="ray")

            num_micro_batches = 1

            # This sets both `torch.manual_seed` and `alpa.manual_seed` (if dist mode) under the hood.
            atorch.manual_seed(seed=123)

            # Assume we have a dataloader that supports `peek` function
            # (i.e. look at next batch but don't advance the pointer)
            pt_inputs, pt_targets = dataloader[0]  # dataloader.peek()

            # Create shape-only version of input
            pt_inputs = torch.empty_like(pt_inputs, device="meta")
            pt_targets = torch.empty_like(pt_targets, device="meta")

            # Functionalize the PyTorch model
            module_func, params, bufs, name_map = atorch.functionalize(
                pt_module,
                [pt_inputs],
                batch_dim=0,
                num_micro_batches=num_micro_batches,
            )

            # Use functional version of optimizer
            # `torch.optim.*` optimizers have in-place ops, so have to implement our own version of optimizer.
            optim_func, optim_state, optim_state_init_func = alpa.torch.optim.adam(params, lr=1e-4)

            # Use functional version of loss function
            loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(*args, **kwargs)

            # Define the training loop
            def sgd_train_func(module_func, loss_func, optim_func, params, bufs, optim_state, batch):
                inputs = batch[0]
                targets = batch[1]

                # wrap forward pass + loss computation in a function
                # @alpa.automatic_layer_construction(layer_num="auto")
                @alpa.automatic_layer_construction(layer_num=2)
                def compute_loss(params, bufs, inputs, targets):
                    # do forward pass
                    bufs, out = module_func(params, bufs, inputs)

                    # do some debugging when in local mode
                    if alpa.torch.mode() == "local":
                        print("out: ", out)

                    # do loss computation
                    loss_value = loss_func(out, targets)
                    return loss_value, bufs

                # do model forward + backward pass
                params_grad, (loss_value, bufs) = atorch.grad_and_value(compute_loss, has_aux=True)(params, bufs, inputs, targets)
                # (loss_value, bufs), params_grad = alpa.value_and_grad(compute_loss, has_aux=True)(params, bufs, inputs, targets)

                # do optimizer step
                params, optim_state = optim_func(params, params_grad, optim_state)

                return params, bufs, optim_state, loss_value

            train_func = functools.partial(sgd_train_func, module_func, loss_func, optim_func)
            parallel_method = alpa.PipeshardParallel(stage_mode="auto", num_micro_batches=num_micro_batches)
            # parallel_method = alpa.PipeshardParallel(stage_mode="auto", num_micro_batches=num_micro_batches, submesh_physical_shape_space="all", submesh_logical_shape_space="all")
            # parallel_method = alpa.PipeshardParallel(stage_mode="uniform", num_micro_batches=num_micro_batches)
            # parallel_method = alpa.ShardParallel(num_micro_batches=num_micro_batches)
            # parallel_method = alpa.ShardParallel()

            def iter_func(params, bufs, optim_state, pt_batch):
                if alpa.torch.mode() == "local":  # local development
                    params, bufs, optim_state, loss_value = train_func(
                        params,
                        bufs,
                        optim_state,
                        pt_batch,
                    )
                    # Show that outputs are in PT format (no need in actual training code)
                    alpa.torch.assert_format("torch", params, bufs, optim_state, loss_value)
                elif alpa.torch.mode() == "dist":  # distributed training
                    params, bufs, optim_state, loss_value = alpa.parallelize(
                        train_func,
                        method=parallel_method,
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
                return params, bufs, optim_state, loss_value

            # Generate sharding plan based on shape-only tensors. Only needed for dist training.
            if atorch.mode() == "dist":
                alpa.global_env.global_config.use_dummy_value_for_benchmarking = True
                params, bufs, optim_state, _ = iter_func(params, bufs, optim_state, (pt_inputs, pt_targets))
                alpa.global_env.global_config.use_dummy_value_for_benchmarking = False

            # Materialize the weights and optimizer state
            params, bufs = weight_init_func(pt_module, name_map, params, bufs)
            optim_state = optim_state_init_func(optim_state)

            # Run training loops
            for i, (pt_inputs, pt_targets) in enumerate(dataloader):
                params, bufs, optim_state, loss_value = iter_func(
                    params, bufs, optim_state, (pt_inputs, pt_targets)
                )
                # do whatever with the loss value, e.g. plot it on a graph
                print("loss value: ", loss_value)

            alpa.shutdown()
