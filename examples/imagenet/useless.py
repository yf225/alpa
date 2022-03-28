
  #sharding_specs = [
  #  pxla.ShardingSpec((pxla.Chunked((8,)), pxla.NoSharding(), pxla.NoSharding(), pxla.NoSharding()),
  #                    (pxla.ShardedAxis(0),)),
  #  pxla.ShardingSpec((pxla.Chunked((8,)),),
  #                    (pxla.ShardedAxis(0),)),
  #]
  #train_iter = create_input_iter(
  #    dataset_builder, local_batch_size, image_size, input_dtype,
  #    sharding_specs, physical_mesh, train=True, cache=config.cache)
  #for i, batch in enumerate(train_iter):
  #    print("driver", batch['image'].shape)
  #    print("driver", batch['label'].shape)
  #    break
  #while True:
  #    pass
  #exit()


    #import tensorflow as tf
    #import numpy as np
    #while True:
    #    yield (np.empty((batch_size, image_size, image_size, 3), np.float32),
    #           np.empty((batch_size,), np.int32))

    import numpy as np
    batch = (np.ones((batch_size, image_size, image_size, 3), np.float32),
             np.ones((batch_size,), np.int32))
    while True:
        yield batch


    #import numpy as np
    #batch = (np.random.randn(batch_size, image_size, image_size, 3).astype(np.float32),
    #         np.random.randn(batch_size).astype(np.int32))
    ##batch = (np.ones((batch_size, image_size, image_size, 3), np.float32),
    ##         np.ones((batch_size,), np.int32))
    #while True:
    #    yield batch


