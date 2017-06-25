from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from keras.backend import tensorflow_backend as tfb
from keras.backend.tensorflow_backend import (
    get_session, is_sparse)

import atexit
atexit.register(tfb.clear_session)
# FIXME: The monkey patch of Function results in an error message at the end of
#     the session. Message:
#         Exception AttributeError: "'NoneType' object has no attribute
#         'TF_DeleteStatus'" in <bound method Session.__del__ of
#         <tensorflow.python.client.session.Session object at 0x7fff6c3b3f90>>
#        ignored
#     Everything seems to work fine though.
#     To inhibit the eror run at the end of main script: KB.clear_session()
#     Workaround is to register clear_session at exit.


__all__ = ('patch',)


# Replacement Function to support enqueue ops.
class Function(object):
    """Runs a computation graph.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        enqueue_ops: List of ops to be run at funtion call for enqueue'ing.
    """

    def __init__(self, inputs, outputs, updates=None, name=None,
                 enqueue_ops=None, **session_kwargs):
        self._enqueue_ops = enqueue_ops or []
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a TensorFlow backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a TensorFlow backend function '
                            'should be a list or tuple.')
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, tuple):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else:
                    # assumed already an op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)
        self.name = name
        self.session_kwargs = session_kwargs

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` should be a list or tuple.')
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            if is_sparse(tensor):
                sparse_coo = value.tocoo()
                indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                          np.expand_dims(sparse_coo.col, 1)),
                                         1)
                value = (indices, sparse_coo.data, sparse_coo.shape)
            feed_dict[tensor] = value
        session = get_session()
        enqueue_ops = self._enqueue_ops
        neops = len(enqueue_ops)
        updated = session.run(enqueue_ops + self.outputs + [self.updates_op],
                              feed_dict=feed_dict,
                              **self.session_kwargs)
        nouts = len(self.outputs)

        # return updated[:len(self.outputs)]
        return updated[neops:nouts + neops]


def patch():
    '''Monkey-patch Keras tensorflow_backend.Function'''
    print('\nMONKEY PATCHING KERAS tensorflow_backend.Function\n',
          file=sys.stderr)
    tfb.Function = Function

