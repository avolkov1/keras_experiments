'''
'''
import os
import pickle
import six
from six.moves import urllib  # @UnresolvedImport
import errno
import tqdm
import inspect

import numpy as np

import logging as logger

from ._dataflow import RNGDataFlow

__all__ = ('Cifar10', 'Cifar100',)

# =============================================================================
# DATASETS
# =============================================================================


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    # y = np.array(y, dtype='int').ravel()
    # if not num_classes:
    #     num_classes = np.max(y) + 1
    # n = y.shape[0]
    # categorical = np.zeros((n, num_classes))
    # categorical[np.arange(n), y] = 1
    categorical = np.zeros(num_classes)
    categorical[y] = 1
    return categorical


_EXECUTE_HISTORY = set()


def execute_only_once():
    """
    Each called in the code to this function is guranteed to return True the
    first time and False afterwards.

    Returns:
        bool: whether this is the first time this function gets called from
            this line of code.

    Example:
        .. code-block:: python

            if execute_only_once():
                # do something only once
    """
    f = inspect.currentframe().f_back
    ident = (f.f_code.co_filename, f.f_lineno)
    if ident in _EXECUTE_HISTORY:
        return False
    _EXECUTE_HISTORY.add(ident)
    return True


def mkdir_p(dirname):
    """ Make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir_, filename=None):
    """
    Download URL to a directory.
    Will figure out the filename automatically from URL, if not given.
    """
    mkdir_p(dir_)
    if filename is None:
        filename = url.split('/')[-1]
    fpath = os.path.join(dir_, filename)

    def hook(t):
        last_b = [0]

        def inner(b, bsize, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return inner
    try:
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) \
                as t:
            fpath, _ = urllib.request.urlretrieve(url, fpath,
                                                  reporthook=hook(t))
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except BaseException:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Download an empty file!"
    # TODO human-readable size
    print('Succesfully downloaded ' + filename + ". " + str(size) + ' bytes.')
    return fpath


DATA_URL_CIFAR_10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_URL_CIFAR_100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def maybe_download_and_extract(dest_directory, cifar_classnum):
    """Download and extract the tarball from Alex's website.
       copied from tensorflow example """
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        cifar_foldername = 'cifar-10-batches-py'
    else:
        cifar_foldername = 'cifar-100-python'
    if os.path.isdir(os.path.join(dest_directory, cifar_foldername)):
        logger.info("Found cifar{} data in {}.".format(cifar_classnum,
                                                       dest_directory))
        return
    else:
        DATA_URL = DATA_URL_CIFAR_10 if cifar_classnum == 10 \
            else DATA_URL_CIFAR_100
        download(DATA_URL, dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        import tarfile
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_cifar(filenames, cifar_classnum, use_categorical=False):
    assert cifar_classnum == 10 or cifar_classnum == 100
    ret = []
    for fname in filenames:
        fo = open(fname, 'rb')
        if six.PY3:
            dic = pickle.load(fo, encoding='bytes')
        else:
            dic = pickle.load(fo)
        data = dic[b'data']
        if cifar_classnum == 10:
            label = dic[b'labels']
            IMG_NUM = 10000  # cifar10 data are split into blocks of 10000
        elif cifar_classnum == 100:
            label = dic[b'fine_labels']
            IMG_NUM = 50000 if 'train' in fname else 10000
        fo.close()
        for k in range(IMG_NUM):
            img = data[k].reshape(3, 32, 32)
            img = np.transpose(img, [1, 2, 0])
            # img /= 255
            lbl = label[k]
            if use_categorical:
                lbl = to_categorical(lbl, cifar_classnum)
            ret.append([img, lbl])
    return ret


def get_filenames(dir_, cifar_classnum):
    assert cifar_classnum == 10 or cifar_classnum == 100
    if cifar_classnum == 10:
        filenames = [os.path.join(
            dir_, 'cifar-10-batches-py', 'data_batch_%d' % i)
            for i in range(1, 6)]
        filenames.append(os.path.join(
            dir_, 'cifar-10-batches-py', 'test_batch'))
    elif cifar_classnum == 100:
        filenames = [os.path.join(dir_, 'cifar-100-python', 'train'),
                     os.path.join(dir_, 'cifar-100-python', 'test')]
    return filenames


def get_dataset_path(*args):
    """
    Get the path to some dataset under ``$TENSORPACK_DATASET``.

    Args:
        args: strings to be joined to form path.

    Returns:
        str: path to the dataset.
    """
    d = os.environ.get('TENSORPACK_DATASET', None)
    if d is None:
        old_d = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'dataflow', 'dataset'))
        old_d_ret = os.path.join(old_d, *args)
        new_d = os.path.join(os.path.expanduser('~'), 'tensorpack_data')
        if os.path.isdir(old_d_ret):
            # there is an old dir containing data, use it for back-compat
            logger.warn("You seem to have old data at {}. This is no longer \
                the default location. You'll need to move it to {} \
                (the new default location) or another directory set by \
                $TENSORPACK_DATASET.".format(old_d, new_d))
        d = new_d
        if execute_only_once():
            logger.warn("$TENSORPACK_DATASET not set, using {} for dataset."
                        .format(d))
        if not os.path.isdir(d):
            mkdir_p(d)
            logger.info("Created the directory {}.".format(d))
    assert os.path.isdir(d), d
    return os.path.join(d, *args)


class CifarBase(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, dir_=None,
                 cifar_classnum=10, use_categorical=True):
        assert train_or_test in ['train', 'test']
        assert cifar_classnum == 10 or cifar_classnum == 100
        self.cifar_classnum = cifar_classnum
        if dir_ is None:
            dir_ = get_dataset_path('cifar{}_data'.format(cifar_classnum))
        maybe_download_and_extract(dir_, self.cifar_classnum)
        fnames = get_filenames(dir_, cifar_classnum)
        if train_or_test == 'train':
            self.fs = fnames[:-1]
        else:
            self.fs = [fnames[-1]]
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test
        self.data = read_cifar(self.fs, cifar_classnum, use_categorical)
        # print('DATA SHAPE: {}'.format(self.data[1].shape))
        self.dir = dir_
        self.shuffle = shuffle

    def size(self):
        return 50000 if self.train_or_test == 'train' else 10000

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            # since cifar is quite small, just do it for safety
            yield self.data[k]

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        fnames = get_filenames(self.dir, self.cifar_classnum)
        all_imgs = [x[0] for x in read_cifar(fnames, self.cifar_classnum)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0, 1))


class Cifar10(CifarBase):
    '''
    Produces [image, label] in Cifar10 dataset,
    image is 32x32x3 in the range [0,255].
    label is an int. If use_categorical then one-hot encoded.
    '''
    def __init__(self, train_or_test, shuffle=True, dir_=None,
                 use_categorical=True):
        '''
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset.
        '''
        # shuffle=True, dir_=None, cifar_classnum=10, use_categorical=True)
        super(Cifar10, self).__init__(
            train_or_test, shuffle=shuffle, dir_=dir_, cifar_classnum=10,
            use_categorical=use_categorical)


class Cifar100(CifarBase):
    '''Similar to Cifar10'''
    def __init__(self, train_or_test, shuffle=True, dir_=None,
                 use_categorical=True):
        super(Cifar100, self).__init__(
            train_or_test, shuffle=shuffle, dir_=dir_, cifar_classnum=100,
            use_categorical=use_categorical)

