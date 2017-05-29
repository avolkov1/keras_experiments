'''
'''
import os
import multiprocessing as mp
import itertools

import msgpack
import msgpack_numpy

import uuid
import zmq

import logging as logger

import traceback

from ._dataflow import ProxyDataFlow
from .concurrency import start_proc_mask_signal


msgpack_numpy.patch()

__all__ = ('PrefetchDataZMQ',)

# =============================================================================
# PREFETCH
# =============================================================================


def dumps(obj):
    """
    Serialize an object.

    Returns:
        str
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads(buf):
    """
    Args:
        buf (str): serialized object.
    """
    return msgpack.loads(buf)


class PrefetchProcessZMQ(mp.Process):
    def __init__(self, ds, conn_name, hwm):
        super(PrefetchProcessZMQ, self).__init__()
        self.ds = ds
        self.conn_name = conn_name
        self.hwm = hwm

    def run(self):
        self.ds.reset_state()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)  # @UndefinedVariable
        self.socket.set_hwm(self.hwm)
        self.socket.connect(self.conn_name)
        while True:
            for dp in self.ds.get_data():
                self.socket.send(dumps(dp), copy=False)


class PrefetchDataZMQ(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZMQ for
    communication.

    A local directory is needed to put the ZMQ pipes.
    You can set this with env var $TENSORPACK_PIPEDIR if you're running on
    non-local FS such as NFS or GlusterFS.

    Note that this dataflow is not fork-safe. You cannot nest this dataflow
    into another PrefetchDataZMQ or PrefetchData.
    """
    def __init__(self, ds, nr_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" for both sender and receiver.
        """
        assert os.name != 'nt', "PrefetchDataZMQ doesn't support windows! "\
            "Consider PrefetchData instead."
        super(PrefetchDataZMQ, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # @UndefinedVariable

        pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
        assert os.path.isdir(pipedir), pipedir
        self.pipename = "ipc://{}/dataflow-pipe-".format(pipedir.rstrip('/')) \
            + str(uuid.uuid1())[:6]
        self.socket.set_hwm(hwm)
        self.socket.bind(self.pipename)

        self.procs = [PrefetchProcessZMQ(self.ds, self.pipename, hwm)
                      for _ in range(self.nr_proc)]
        self.start_processes()
        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def start_processes(self):
        start_proc_mask_signal(self.procs)

    def get_data(self):
        try:
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                dp = loads(self.socket.recv(copy=False).bytes)
                yield dp
        except zmq.ContextTerminated:
            logger.info("ContextTerminated in Master Prefetch Process")
            return
        except BaseException:
            raise

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass

    def __del__(self):
        # on exit, logger may not be functional anymore
        if not self.context.closed:
            self.context.destroy(0)
        for x in self.procs:
            x.terminate()
        try:
            # TODO test if logger here would overwrite log file
            print("Prefetch process exited.")
        except BaseException:
            traceback.print_exc()
            pass

