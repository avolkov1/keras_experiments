'''
Useful utility functions/classes.
'''
import sys

try:
    from cStringIO import StringIO
except ImportError:
    # Python 3 compat.
    from io import StringIO


__all__ = ('Capturing', 'dummy_context_mgr',)


# noqa ref: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list):
    '''
    Capture stdout into a list like object. Ex:
        with Capturing() as msum:
            minfo = submodel.summary()
            print('\t{}\n\t{}\n'.format('\n\t'.join(msum), minfo))
    '''
    def __init__(self):
        super(Capturing, self).__init__()
        self._stdout = sys.stdout
        self._stringio = StringIO()

    def __enter__(self):
        # self._stdout = sys.stdout
        # self._stringio = StringIO()
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):  # @UnusedVariable
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# noqa ref: https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
class dummy_context_mgr(object):  # pylint: disable=invalid-name
    '''
    Use contexts/with conditionally. Ex:
      with tf.device(gpu0) if ngpus > 1 else dummy_context_mgr():  # @NoEffect
    '''
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):  # @UnusedVariable
        return False
