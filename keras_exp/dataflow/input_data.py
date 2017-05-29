'''
'''
import logging as logger
import tensorflow as tf

from .concurrency import ShareSessionThread


# TODO enqueu_many? https://github.com/tensorflow/tensorflow/issues/7817#issuecomment-282053155 @IgnorePep8
class EnqueueThread(ShareSessionThread):
    def __init__(self, queue, ds, input_placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread'
        self.daemon = True

        self.dataflow = ds
        self.queue = queue

        self.placehdrs = input_placehdrs

        self.op = self.queue.enqueue(self.placehdrs)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)
        self.size_op = self.queue.size()
        # add_moving_summary(tf.cast(
        #     self.size_op, tf.float32, name='input_queue_size'))

    def run(self):
        with self.default_sess():
            try:
                self.dataflow.reset_state()
                while True:
                    for dp in self.dataflow.get_data():
                        feed = dict(zip(self.placehdrs, dp))
                        # print 'qsize:', self.sess.run(
                        #     [self.op, self.size_op], feed_dict=feed)[1]
                        self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("EnqueueThread Exited.")

