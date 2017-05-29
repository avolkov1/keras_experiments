'''
'''
from abc import abstractmethod, ABCMeta
import copy as copy_mod
import logging as logger
import six

from ._dataflow import (MapDataComponent, get_rng)

__all__ = ('AugmentImageComponent', 'ImgScale')


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.
        """
        # d, params = self._augment_return_params(d)
        d, _ = self._augment_return_params(d)
        return d

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    @abstractmethod
    def _augment(self, d, param):
        """
        augment with the given param and return the new image
        """

    def _get_augment_params(self, d):
        """
        get the augmentor parameters
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)


class ImageAugmentor(Augmentor):
    def _fprop_coord(self, coord, param):
        return coord


class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be
                applied.
        """
        self.augs = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        # the next augmentor requires the previous one to finish
        raise RuntimeError("Cannot simply get parameters of a AugmentorList!")

    def _augment_return_params(self, img):
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augs:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augs, param):
            img = aug._augment(img, prm)
        return img

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augs:
            a.reset_state()


class AugmentImageComponent(MapDataComponent):
    """
    Apply image augmentors on 1 component.
    """
    def __init__(self, ds, augmentors, index=0, copy=True):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of
                :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int): the index of the image component to be augmented.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
        """
        if isinstance(augmentors, AugmentorList):
            self.augs = augmentors
        else:
            self.augs = AugmentorList(augmentors)

        self._nr_error = 0

        def func(x):
            try:
                if copy:
                    x = copy_mod.deepcopy(x)
                ret = self.augs.augment(x)
            except KeyboardInterrupt:
                raise
            except Exception:
                self._nr_error += 1
                if self._nr_error % 1000 == 0 or self._nr_error < 10:
                    logger.exception("Got {} augmentation errors."
                                     .format(self._nr_error))
                return None
            return ret

        super(AugmentImageComponent, self).__init__(
            ds, func, index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs.reset_state()


class ImgScale(ImageAugmentor):
    def __init__(self, scale=255.0):
        self._scale = scale
        super(ImgScale, self).__init__()

    def _augment(self, img, _):
        return img / self._scale

