from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from glob import glob

PACKAGE_NAME = 'dicelib'
PACKAGE_VERSION = '1.0.0'
PACKAGE_DESCRIPTION = 'Software library of the Diffusion Imaging and Connectivity Estimation (DICE) lab'

def get_extensions():
    # ext1 = Extension(name='commit.trk2dictionary',
    #                  sources=['commit/trk2dictionary/trk2dictionary.pyx'],
    #                  extra_compile_args=['-w'],
    #                  language='c++')

    # ext2 = Extension(name='commit.core',
    #                  sources=['commit/core.pyx'],
    #                  extra_compile_args=['-w'],
    #                  language='c++')

    # ext3 = Extension(name='commit.proximals',
    #                  sources=['commit/proximals.pyx'],
    #                  extra_compile_args=['-w'],
    #                  language='c++')

    return []#[ext1, ext2, ext3]


class CustomBuildExtCommand(build_ext):
    """build_ext command to use when numpy headers are needed"""

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include

        # Add everything requires for build
        self.swig_opts = None
        self.include_dirs = [get_include()]
        self.distribution.ext_modules[:] = cythonize(
                    self.distribution.ext_modules)

        # Call original build_ext command
        build_ext.finalize_options(self)
        build_ext.run(self)


setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    description=PACKAGE_DESCRIPTION,
    long_description=PACKAGE_DESCRIPTION,
    author='Alessandro Daducci',
    author_email='alessandro.daducci@univr.it',
    packages=find_packages(),
    cmdclass={'build_ext': CustomBuildExtCommand},
    ext_modules=get_extensions(),
    setup_requires=['Cython>=0.29', 'numpy>=1.12'],
    install_requires=['wheel', 'setuptools>=46.1', 'numpy>=1.12', 'scipy>=1.0', 'Cython>=0.29', 'dipy>=1.0'],
    scripts=glob('bin/*.py')
)
