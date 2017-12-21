'''
'''
import sys

try:
    from cStringIO import StringIO
except ImportError:
    # Python 3 compat.
    from io import StringIO


__all__ = ('Capturing',)


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
