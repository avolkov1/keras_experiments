

from ._multigpu import *

# If importing wildcard rename optimizers to _optimizers. Otherwise import
# into optimizers class namespace as done below in class optimizers.
# from .optimizers import *

# from importlib import import_module as _im  # @IgnorePep8
#
#
# class optimizers(object):
#     from .optimizers import __all__ as _optlist
#     for _obj in _optlist:
#         locals()[_obj] = getattr(_im('keras_exp.multigpu.optimizers'), _obj)
#
#     del _obj
#     del _optlist
#
#
# del _im

