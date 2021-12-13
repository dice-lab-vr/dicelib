from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from glob import glob
from numpy import get_include


def get_extensions():
    processing = Extension(
        name='dicelib.tractogram',
        sources=['dicelib/tractogram.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    lazytck = Extension(
        name='dicelib.lazytck',
        sources=['dicelib/lazytck.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language="c++"
    )

    return [ processing, lazytck ]


class CustomBuildExtCommand(build_ext):
    """build_ext command to use when numpy headers are needed"""

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include

        # Add everything requires for build
        self.swig_opts = None
        self.include_dirs = [get_include()]
        self.distribution.ext_modules[:] = cythonize(self.distribution.ext_modules)

        # Call original build_ext command
        build_ext.finalize_options(self)
        build_ext.run(self)

import dicelib.info as info
setup(
    name=info.NAME,
    version=info.VERSION,
    description=info.DESCRIPTION,
    long_description=info.LONG_DESCRIPTION,
    author=info.AUTHOR,
    author_email=info.AUTHOR_EMAIL,
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=get_extensions(),
    packages=find_packages(),
    setup_requires=['Cython>=0.29', 'numpy>=1.12'],
    install_requires=['wheel', 'setuptools>=46.1', 'numpy>=1.12', 'scipy>=1.0', 'Cython>=0.29', 'dipy>=1.0'],
    scripts=glob('bin/*.py'),
    zip_safe=False
)
