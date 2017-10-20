from setuptools import setup, find_packages

_pkgname = 'keras_exp'
packages_list = [_pkgname] + \
    ['{}.{}'.format(_pkgname, subpkg)
     for subpkg in find_packages(where=_pkgname)]

setup(
    name='keras_exp',
    version='1.0',
    description='Experimental Keras libraries and examples.',
    author='Alex Volkov',
    url='https://github.com/avolkov1/keras_experiments',
    packages=packages_list,
    license='unlicense.org',
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Unlicense',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ]
)
