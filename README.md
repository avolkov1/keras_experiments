# Keras Experiments
Experimental Keras libraries and examples. The `keras_exp` package is for
exploring experimental and new features of Keras. Primary focus is on using
Keras in conjuction with Tensorflow for multi-GPU and distributed systems.
Currently this package is not hosted on PyPI. My goals here are to illustrate
and explore APIs for multi-GPU and distributed systems. If installing I would
recommend installing this package in editable mode:

```
git clone https://github.com/avolkov1/keras_experiments.git
cd keras_experiments
pip install -e .  # preferably in a virtualenv
```

or install directly from github:
```
pip install -e git+https://github.com/avolkov1/keras_experiments.git#egg=keras_exp
```

Otherwise use `PYTHONPATH`, symlinks, etcetera to use the package within your
python environment. Feel free to modify, copy, and do whatever you like with
the code.


#### Examples
In general you should run latest version of Keras for various examples. The
latest release as of this update is Keras 2.1.1. This release has many great
features. I will be cleaning up the code in the `keras_exp` package relying
on features in the mainline Keras. For example, the monkey patch for the
backend TF function is no longer necessary given the ability to pass fetches to
K.Function() with the TensorFlow backend (Keras 2.1.0 improvement). Even the
`make_parallel` function itself is no longer necessary as it has been
incorporated in Keras 2.0.9 `keras.utils.multi_gpu_model`. A few enhancements
remain. Refer to [`make_parallel`](#function-make_parallel) below.

* [mnist\_tfrecord\_mgpu.py](examples/mnist/mnist_tfrecord_mgpu.py)

    Mnist example with multi-GPU data parallelism distribution using TFRecords
    and TF queue pipeline to run Keras. Requires Keras v 2.0.8.
    Run it as follows:
    ```bash
    cd mnist/examples
    CUDA_VISIBLE_DEVICES=0 python mnist_tfrecord_mgpu.py  # 1 - GPU
    CUDA_VISIBLE_DEVICES=0,1 python mnist_tfrecord_mgpu.py  # 2 - GPUs
    ```
    Avoiding feed_dict and using a TF queue can give some performance
    improvements.
    Original implementation of this can be found here:
    [https://github.com/fchollet/keras/blob/master/examples/mnist_tfrecord.py](https://github.com/fchollet/keras/blob/master/examples/mnist_tfrecord.py)


* [cifar10\_cnn\_mgpu.py](examples/cifar/cifar10_cnn_mgpu.py)

    Cifar10 example with multi-GPU options. Run it as follows:
    ```bash
    python examples/cifar/cifar10_cnn_mgpu.py --help # read instructions
    # Use CUDA_VISIBLE_DEVICES to mask GPUs from Tensorflow otherwise uses all.
    CUDA_VISIBLE_DEVICES=0,1,2 python \
        examples/cifar/cifar10_cnn_mgpu.py --mgpu --epochs=10 --checkpt
    ```

The `*_tfqueue` examples use preloaded API. Refer to:

[https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data](https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data)

The preloaded API for numpy arrays has been superseded by TF's Dataset API, but
sometimes (such as some sophisticated data augmentation case) it is still
useful to implement it via low level. For Dataset API refer to:

[https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays](https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays)

Keras example via Dataset API can be found here:

[https://github.com/fchollet/keras/blob/master/examples/mnist_dataset_api.py](https://github.com/fchollet/keras/blob/master/examples/mnist_dataset_api.py)

* [cifar10\_cnn\_mgpu_tfqueue.py](examples/cifar/cifar10_cnn_mgpu_tfqueue.py)

    Cifar10 example with multi-GPU options using Tensorflow queue. This is a
    similar example to `mnist_tfrecord_mgpu.py`. Run it as follows:
    ```bash
    python examples/cifar/cifar10_cnn_mgpu_tfqueue.py --help # read instructions
    # Use CUDA_VISIBLE_DEVICES to mask GPUs from Tensorflow otherwise uses all.
    # If CUDA_VISIBLE_DEVICES=0 or a single GPU, the --mgpu flag is ignored.
    CUDA_VISIBLE_DEVICES=0 python \
        examples/cifar/cifar10_cnn_mgpu_tfqueue.py --mgpu --epochs=10 # single GPU 
    CUDA_VISIBLE_DEVICES=0,1,2 python \
        examples/cifar/cifar10_cnn_mgpu_tfqueue.py --mgpu --epochs=10 # multi-GPU
        
    ```
    Use Tensorflow version 1.2.x and above for good performance. Runs slightly
    faster than `cifar10_cnn_mgpu.py`, especially when using augmentation.

* [cifar10\_cnn\_mgpu_tfqueue\_var.py](examples/cifar/cifar10_cnn_mgpu_tfqueue_var.py)

    Similarly to `cifar10_cnn_mgpu_tfqueue.py` example, this is a slightly
    different approach on using preloaded data in a Tensorflow queue via
    placeholder and `tf.Variable`. Run the same way as
    `cifar10_cnn_mgpu_tfqueue.py` example.

The examples below are implemented via Horovod.

[https://eng.uber.com/horovod/](https://eng.uber.com/horovod/)

Refer to the bottom of this README for setting up Horovod. Compare these
implemenations to the ones using `make_parallel`. Main difference is that
Horovod implements synchronous training, while the `make_parallel` implements
asynchronous training.

* [cifar10\_cnn\_horovod.py](examples/cifar/cifar10_cnn_horovod.py)

    Using 4 GPUs with data augmentation run the example via:
    ```
    TMPDIR=/tmp mpirun --report-bindings --map-by ppr:2:socket -oversubscribe \
        -np 4 python2 ./examples/cifar/cifar10_cnn_horovod.py --epochs=4 --aug
    ```

    To change the number of GPUs change the option `-np`.

* [cifar10\_cnn\_horovod\_tfqueue.py](examples/cifar/cifar10_cnn_horovod_tfqueue.py)

    Same as the `cifar10_cnn_horovod.py`, but using the TF queue pipeline.

The Horovod framework seems to scale extremely well. It enables one to also
easily scale multinode. The examples above can easily be scaled to multinode
by changing `--np 8` assuming 4 GPUs per node. Tensorflow/Keras code on HPC
type networks with NCCL2 and Horovod is very easy to scale. Additional
enhancements such as RDMA and/or GPUDirect RDMA are transparently used via
MPI+NCCL2 combination implemented in Horovod.


#### Usage
Refer to the example above for detailed usage. Typical usage is to define
a Keras model and then call the model conversion function or class to make it
run on multiple GPUs.

##### Function: make\_parallel
A similar function has been added to mainstream Keras version 2.0.9
`keras.utils.multi_gpu_model`:<br/>
[https://github.com/fchollet/keras/releases/tag/2.0.9](https://github.com/fchollet/keras/releases/tag/2.0.9)

The version implemented here differs slightly. The slicing is done on CPU
which enables using batch sizes that exceed a single GPUs memory. Also added
ability to save/load of parameters of original serial model with multigpu model
instance via `ModelGPU` class wrapper.
Refer to this comment:<br/>
[https://github.com/fchollet/keras/issues/2436#issuecomment-337591230](https://github.com/fchollet/keras/issues/2436#issuecomment-337591230)

There is also an attempt in this version to implement synchronized training via
gradient synchronization. Synchronized training doesn't seem to work properly
yet so contributions are welcome. Another feature is to use StagingArea to
speedup data copy to GPUs. Refer to issue #2 as this is in the works.

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

Note, for some reason often the validation loss and accuracy reported during
training by Keras is incorrect when the batch is sliced per `make_parallel`
function. Therefore I would recommend that after training a parallel model run
validation with the serial model.
```python
# EXAMPLE: Validate after training to get the correct loss (accuracy) measure.
serial_model.compile(loss='categorical_crossentropy',
                     optimizer=opt, metrics=['accuracy'])
metrics = serial_model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
print('\nVALIDATION LOSS, ACC: {}, {}'.format(*metrics))
```


#### Distributed
I would like to note that at this time Horovod is probably the easiest approach
to adopt for multinode scaling. The approach below might still be beneficial
if MPI is not available.

Familiarize yourself with basics of Tensorflow distributed.

[https://www.tensorflow.org/extend/architecture](https://www.tensorflow.org/extend/architecture)

[https://www.tensorflow.org/deploy/distributed](https://www.tensorflow.org/deploy/distributed)

The current ```keras_exp.distrib``` implementation corresponds to
"Between-graph replication" and "Asynchronous training".

The distributed examples below are for SLURM setups. To implement for another
cluster type one would need to implement a concrete class derived from
```keras_exp.distrib.cluster_parsers.base.ClusterParser``` class. I would like
to add other clusters in the future such as Spark standalone clusters. SLURM
like clusters i.e. PBS, UGE, etcetera, should have similar implementation to the
SLURM ```keras_exp.distrib.cluster_parsers.slurm.SlurmClusterParser``` parser.
I might add a generic mpi4py based cluster parser as well.

The examples that follow are SLURM specific, but should be straightforward to
adopt for other cluster types. Start with a SLURM interactive session (adopting
for sbatch is left to the user). The sessions should be started via either of
the two following approaches:

APPROACH 1 RECOMMENDED:
[1 Parameter Server + 1 Worker (that could use one or multiple GPU devices)] PER NODE
```bash
srun -N $NNODES_DESIRED --ntasks-per-node=2 -p gpu_queue --pty bash
```

APPROACH 2 WORKS BUT NOT AS PERFORMANT THEREFORE NOT RECOMMENDED:
[1 Parameter Server + N Workers (1 GPU device PER WORKER)] PER NODE
```bash
srun -N $NNODES_DESIRED --ntasks-per-node=${NWORKERS_DESIRED_PLUS_ONE} -p gpu_queue --pty bash
```

Approach 1 is typically better in performance. On a given node the worker uses
all the GPUs available on the node. If you desire to limit number of GPUs in
this scenario, use the ```CUDA_VISIBLE_DEVICES``` environment var. In general
I assume that there are the same number of GPUs on the
nodes i.e. if node 1 has 4 GPUs then it is assumed node 2 has 4 GPUs also.
Example v2 below though should work with varying number of GPUs on the nodes,
but I did not test it in that scenario.

Once on the node run test via:
```bash
srun python -m keras_exp.distrib.slurm
```

##### Examples

* [cifar10\_cnn\_distrib\_slurm.py](examples/cifar/cifar10_cnn_distrib_slurm.py)

    Cifar10 example experimentation with distribution on a SLURM cluster. The
    run command would be something similar to:
    ```bash
    srun python examples/cifar/cifar10_cnn_distrib_slurm.py
    ```

* [cifar10\_cnn\_distrib\_v2\_slurm.py](examples/cifar/cifar10_cnn_distrib_v2_slurm.py)

    Cifar10 version 2 example experimentation with distribution on a SLURM
    cluster. Prior to running this example pre-download the data so that
    multiple processes are not downloading over each other. Suggestion: run the
    ```cifar10_cnn_mgpu.py``` to pre-download the data. The keras Cifar10
    data is downloaded under ```$HOME/.keras/datasets``` location.
    The run command would be something similar to:
    ```bash
    srun python examples/cifar/cifar10_cnn_distrib_v2_slurm.py
    ```

The difference in the two distributed Cifar examples above is that in the first
example, all the workers join the server except for chief worker. The chief
worker then distributes the computation graph to all devices that are
distributed on different workers. The inefficiency with this approach is that
data is read on the chief's node and communicated to other nodes if the devices
are on the other nodes. This first example implements data-parallelism across
devices, but not nodes.

The v2 example the workers are not joined. Instead each worker loads the data
independently, and runs the computation on devices it directly connects to i.e.
devices are on the same host as the worker. This avoids communicating data from
chief node to other nodes. This 2nd version also runs a subset of the overall
data i.e. a random slice of data correspondsing to its rank number, hence
implementing data-parallelism on the node level as well as device level.

Benchmarking 1 run comparison running on 2 Nodes APPROACH 1 i.e. 1 PS and 1
Worker per node with 4 P100 GPUs on each node. Total of 8 GPUs.

NO RDMA:

```bash

time srun python ./examples/cifar/cifar10_cnn_distrib_slurm.py --epochs=3

ea. epoch runtime: 54s
achieved: val_acc: 0.6021
walltime: 3m5.667s

time srun python ./examples/cifar/cifar10_cnn_distrib_v2_slurm.py --epochs=3

ea. epoch runtime: 49s
achieved: val_acc: 0.5876
walltime: 2m53.116s
```

WITH RDMA:
```bash

time srun python ./examples/cifar/cifar10_cnn_distrib_slurm.py --epochs=3 --rdma

ea. epoch runtime: 6s
achieved: val_acc: 0.5749
walltime: 0m39.046s

time srun python ./examples/cifar/cifar10_cnn_distrib_v2_slurm.py --epochs=3 --rdma

ea. epoch runtime: 5s
achieved: val_acc: 0.5875
walltime: 0m34.461s
```

The performance above does not vary too much between the two examples, but
bear in mind that Cifar10 is not a very stressful benchmark. It does not even
make sense to run it distributed per se. This was done for example purposes to
demonstrate the code API. Do note though, that RDMA makes a substantial difference.
The cluster above has RDMA between the nodes. The data itself resides on NFS.
I think that in a larger scale scenario, version 2 example's approach should
scale better.

Disclaimer: The distributed implementation is still very beta and needs more
testing. I am still learning and trying to understand distributed Tensorflow.

##### RDMA with Tensorflow
Tensorflow version 1.3+ have the RDMA fixes incorporated, so no need to patch
with TF 1.3+ versions. Also, use Bazel 0.5.4. Instructions that follow are for
TF 1.2.1.

In order to use RDMA I compiled Tensorflow 1.2.1 with a patch.

```bash
# install bazel
mkdir -p ~/progs/bazel && cd ~/progs/bazel
wget https://github.com/bazelbuild/bazel/releases/download/0.5.2/bazel-0.5.2-installer-linux-x86_64.sh
chmod u+x bazel-0.5.2-installer-linux-x86_64.sh
./bazel-0.5.2-installer-linux-x86_64.sh --user

# clone tensorflow v1.2.1
mkdir -p ~/gitrepos && cd ~/gitrepos
git clone https://github.com/tensorflow/tensorflow -b v1.2.1
cd tensorflow
git branch v1.2.1_patchrdma
git fetch origin master
git cherry-pick 314008aceb061675eb30daf7e0cae303ebfe8723
# https://github.com/tensorflow/tensorflow/commit/314008aceb061675eb30daf7e0cae303ebfe8723
```

Then just compile Tensorflow per official instructions. On my cluster I do it
like this:
```bash
module load PrgEnv/GCC+OpenMPI/2017-03-20  cuda/8.0.44  cudnn/cuda80-v51

mkvirtualenv py-tfrdma
pip install -U pip
pip install wheel numpy
# pip install other stuff like keras with --no-deps etc.

./configure
# answer a bunch of questions. For RDMA:
# Do you wish to build TensorFlow with VERBS support? [y/N] y

# compile (first install bazel)
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

# after finished run:
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp/tensorflow_pkg

# pip installing into py-tfrdma virtualenv 
pip install $HOME/tmp/tensorflow_pkg/tensorflow-1.2.1-cp27-cp27mu-linux_x86_64.whl

```


#### Dockerfiles with Horovod
Horovod is a "distributed training framework for TensorFlow" that has been open
sourced by Uber. It can also be used within a node for effective multigpu
parallelization or distributed across nodes. Please refer to horovod github for
details:

[https://github.com/uber/horovod](https://github.com/uber/horovod)

I added two Dockerfiles for setting up Horovod. These containers can be used
for straightforward parallelization within a single node.

* [Dockerfile\_tfnvcr\_horovod](dockerfiles/Dockerfile_tfnvcr_horovod)

    This dockerfile is based off of Tensorflow containers found on the Nvidia
    GPU Cloud Registry. [https://ngc.nvidia.com](https://ngc.nvidia.com).
    Go to the site to register and obtain an API key for docker login.
    Login to registry via:
    ```
    docker login nvcr.io -u \$oauthtoken -p <API_KEY_FROM_NGC>
    ```
    NCCL2 is already installed in Tensorflow containers from nvcr.io

* [Dockerfile\_tfpub\_horovod](dockerfiles/Dockerfile_tfpub_horovod)

    This dockerfile is based off of Tensorflow containers found on docker hub.
    Download NCCL2 from:<br/>
    &emsp;&emsp;[https://developer.nvidia.com/nccl/nccl-download](https://developer.nvidia.com/nccl/nccl-download)<br/>
    When building this container set the path to the deb archive.
    ```
    docker build \
        --build-arg BASECONTAINER=tensorflow/tensorflow:1.3.0-devel-gpu \
        --build-arg NCCL2DEBFILE=local_path_to_NCCL2_debfile \
        -t tf1.3.0_horovod -f ./dockerfiles/Dockerfile_tfpub_horovod .
    ```
    Change basecontainer, tag, nccl2 file as needed. Current NCCL2 deb files
    are:
    ```
    nccl-repo-ubuntu1404-2.0.5-ga-cuda8.0_2-1_amd64.deb
    nccl-repo-ubuntu1404-2.0.5-ga-cuda9.0_3-1_amd64.deb
    nccl-repo-ubuntu1604-2.0.5-ga-cuda8.0_2-1_amd64.deb (this one works with tensorflow/tensorflow:1.3.0-devel-gpu)
    nccl-repo-ubuntu1604-2.0.5-ga-cuda9.0_3-1_amd64.deb
    ```
    Use the appropriate NCCL2 for the corresponding Tensorflow base container.

I will be adding Horovod examples for comparison. Horovod uses synchronous
training which compared to asynchronous training generally converges faster
during training.
