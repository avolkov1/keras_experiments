'''
'''
from contextlib import contextmanager
import signal
import threading
import logging as logger

import keras.backend as KB

# =============================================================================
# CONCURRENCY: THREADS, PROCS
# =============================================================================


@contextmanager
def mask_sigint():
    """
    Returns:
        a context where ``SIGINT`` is ignored.
    """
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    yield
    signal.signal(signal.SIGINT, sigint_handler)


def start_proc_mask_signal(proc):
    """ Start process(es) with SIGINT ignored.

    Args:
        proc: (multiprocessing.Process or list)
    """
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for p in proc:
            p.start()


class ShareSessionThread(threading.Thread):
    """ A wrapper around thread so that the thread
        uses the default session at "start()" time.
    """
    def __init__(self, th=None):
        """
        Args:
            th (threading.Thread or None):
        """
        super(ShareSessionThread, self).__init__()
        if th is not None:
            assert isinstance(th, threading.Thread), th
            self._th = th
            self.name = th.name
            self.daemon = th.daemon

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warn('ShareSessionThread {} wasn\'t under a default '
                        'session!'.format(self.name))
            yield

    def start(self):
        # import tensorflow as tf
        # self._sess = tf.get_default_session()
        self._sess = KB.get_session()
        super(ShareSessionThread, self).start()

    def run(self):
        if not self._th:
            raise NotImplementedError()
        with self._sess.as_default():
            self._th.run()

