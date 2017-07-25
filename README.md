# Keras Experiments
Experimental Keras libraries and examples.


#### Examples
* [mnist\_tfrecord\_mgpu.py](examples/mnist/mnist_tfrecord_mgpu.py)

    Mnist example with multi-GPU data parallelism distribution using TFRecords
    and TF queue pipline to run Keras. Run it as follows:
    ```bash
    cd mnist/examples
    CUDA_VISIBLE_DEVICES=0 python mnist_tfrecord_mgpu.py  # 1 - GPU
    CUDA_VISIBLE_DEVICES=0,1 python mnist_tfrecord_mgpu.py  # 2 - GPUs
    ```
    Avoiding feed_dict and using a TF queue can give significant performance
    improvements. I will try to add a corresponding Cifar10 example with this
    paradigm.
    Original implementation of this can be found here:
    [https://github.com/fchollet/keras/pull/7075](https://github.com/fchollet/keras/pull/7075)


* [cifar10\_cnn\_mgpu.py](examples/cifar/cifar10_cnn_mgpu.py)

    Cifar10 example with multi-GPU options. Run it as follows:
    ```bash
    python examples/cifar/cifar10_cnn_mgpu.py --help # read instructions
    # Use CUDA_VISIBLE_DEVICES to mask GPUs from Tensorflow otherwise uses all.
    CUDA_VISIBLE_DEVICES=0,1,2 python \
        examples/cifar/cifar10_cnn_mgpu.py --mgpu --epochs=10 --checkpt
    ```

* [cifar10\_cnn\_distrib\_slurm.py](examples/cifar/cifar10_cnn_distrib_slurm.py)

    Cifar10 example experimentation with distribution on a SLURM cluster. The
    run command would be something similar to:
    ```bash
    srun python examples/cifar/cifar10_cnn_distrib_slurm.py
    ```
    Run a test via:
    ```bash
    srun python -m keras_exp.distrib.slurm
    ```

#### Usage
Refer to the example above for detailed usage. Typical usage is to define
a Keras model and then call the model conversion function or class to make it
run on multiple GPUs.

##### Function: make\_parallel
```python
def make_parallel(serial_model, gdev_list, ps_device='/cpu:0', usenccl=False,
                  initsync=True, syncopt=False, enqueue=False,
                  model_class=ModelMGPU):
    '''Given a keras model, return an equivalent model which parallelizes
    the computation over multiple GPUs listed in the gdev_list.

    Data-Parallel:
    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.

    If getting an out-of-memory (OOM) error when scaling the batch size by the
    number of GPUs, there might be input layer(s) in the serial model that runs
    additional special operations (such as tranformation of some sort) on the
    1st GPU as enumerated by Tensorflow. This was an observed behavior for
    Embedding layers. The workaround is to pin such layers to the CPU, or
    simply pin the instantiation of the serial mode to CPU. The parallelization
    will move the operations to GPU.

    :Example:

        if mgpu_flag:
            with tf.device('/cpu:0'):
                # define the serial model.
                model_serial = get_model_serial()

            gdev_list = get_available_gpus()
            model = make_parallel(model_serial, gdev_list)
        else:
            model = def_model_serial()

    :param Model serial_model: Serial i.e. non-multi GPU Keras model.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.
        This can be a list of strings or list of instances of tf.DeviceSpec.

    :param str ps_device: Parameter server device to use.

    :param bool usenccl: Use the contrib.nccl Tensorflow library for initial
        parameter synchronization and gradients averaging. Note, the model's
        usenccl option overrides the optimizers usenccl option.
        Default: False

    :param bool initsync: Synchronize initial Variables i.e. weights,
        biases, etc. Default: True

    :param bool syncopt: Synchronize gradients. Requires a multi-gpu optimizer.
        Default: False

    :param bool enqueue: Use StagingArea in the multi-GPU model. Could
        potentially speed up Host-to-Device transfers.
        Produces a warning that kwargs are ignored for Tensorflow. The
        _patch_tf_backend module mokey patches the Function in
        tensorflow_backend to use the enqueue_ops option.
        Default: False

    :param model_class: Class object to instantiate for multi-gpu models. This
        is needed when the ModelMGPU is mixed-in with other classes.
        Default: ModelMGPU

    :returns: Multi-GPU parallelized model. If ngpus < 2 then do nothing and
        return the provided serial_model.
    :rtype: ModelMGPU
    '''
```

Usage summary:

```python
# EXAMPLE
from keras_exp.multigpu import get_available_gpus
from keras_exp.multigpu import make_parallel
# or using class # from keras_exp.multigpu import ModelMGPU

# serial_model = ... # some Keras model
gdev_list = get_available_gpus()
mgpu_model = make_parallel(serial_model, gdev_list)
# or using class
# mgpu_model = ModelMGPU(serial_model=serial_model, gdev_list=gdev_list)
# ... Setup optimizer, compile, etc. Run training.
# mgpu_model.compile(...)
# mgpu_model.fit(...)
```

Unles you are experimenting and tinkering with code under the hood leave the
following parameters at their defaults:
```usenccl=False, initsync=True, syncopt=False, enqueue=False,```

Maybe set initsync to False as it should not make a difference.

