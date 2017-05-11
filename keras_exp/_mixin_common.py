'''
'''
import inspect

__all__ = ('mixedomatic',)


# ref: http://stackoverflow.com/a/6100595/3457624
def mixedomatic(ignore_kargs_spec=False):
    """Mixed-in class decorator.

    Must specify arguments as keywords. Calls the __init__ of all classes
    according to reverse(__bases__) which should match mro. The kwargs are
    matched to each class. If a particular class accepts **kwargs then kwargs
    are passed through unless ignore_kargs_spec is True.
    """
    def mixedomatic_(cls):
        classinit = cls.__init__ if '__init__' in cls.__dict__ else None

        def getargs(aspec, kwargs):
            """Get key-word args specific to the args list in the args spec.

            :param aspec: Argspec returned by inspect.getargspec.
            :type aspec: :class:`inspect.ArgSpec`
            :param dict kwargs: Dictionary of keyword arguments.
            """
            if aspec.keywords is not None and not ignore_kargs_spec:
                return kwargs

            _kwargs = {iarg: kwargs[iarg]
                       for iarg in aspec.args if iarg in kwargs}
            return _kwargs

        # define an __init__ function for the class
        def __init__(self, **kwargs):
            # call the __init__ functions of all the bases
            for base_ in reversed(cls.__bases__):
                aspec = inspect.getargspec(base_.__init__)
                base_kwargs = getargs(aspec, kwargs)
                base_.__init__(self, **base_kwargs)

            # also call any __init__ function that was in the class
            if classinit:
                aspec = inspect.getargspec(cls.__init__)
                _kwargs = getargs(aspec, kwargs)
                classinit(self, **_kwargs)

        # make the local function the class's __init__
        setattr(cls, '__init__', __init__)
        return cls

    return mixedomatic_

