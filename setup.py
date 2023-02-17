from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext
from glob import glob
from numpy import get_include
import shutil

# name of the package
package_name = 'dicelib'


def get_extensions():
    lazytractogram = Extension(
        name='lazytractogram',
        sources=[f'{package_name}/lazytractogram.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    image = Extension(
        name='image',
        sources=[f'{package_name}/image.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    streamline = Extension(
        name='streamline',
        sources=[f'{package_name}/streamline.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    tractogram = Extension(
        name='tractogram',
        sources=[f'{package_name}/tractogram.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    clustering = Extension(
        name='clustering',
        sources=[f'{package_name}/clustering.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    split_cluster = Extension(
        name='split_cluster',
        sources=[f'{package_name}/split_cluster.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11'],
        language='c++',
    )
    return [ lazytractogram, image, streamline, tractogram, clustering, split_cluster ]


class CustomBuildExtCommand(build_ext):
    """build_ext command to use when numpy headers are needed"""

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include

        # Add everything requires for build
        self.swig_opts = None
        self.include_dirs = [get_include()]
        self.distribution.ext_modules[:] = cythonize(self.distribution.ext_modules, build_dir='build')
        print( self.distribution.ext_modules )

        # Call original build_ext command
        build_ext.finalize_options(self)
        build_ext.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        shutil.rmtree('./build')


# import details from {package_name}/info.py
import sys
sys.path.insert(0, f'./{package_name}/')
import info

setup(
    name=info.NAME,
    version=info.VERSION,
    description=info.DESCRIPTION,
    long_description=info.LONG_DESCRIPTION,
    author=info.AUTHOR,
    author_email=info.AUTHOR_EMAIL,
    cmdclass={
        'build_ext': CustomBuildExtCommand,
        'clean': CleanCommand
    },
    ext_package=package_name,
    ext_modules=get_extensions(),
    packages=find_packages(),
    setup_requires=['wheel', 'Cython>=0.29', 'numpy>=1.12'],
    install_requires=['setuptools>=46.1', 'numpy>=1.12', 'scipy>=1.0', 'cython>=0.29', 'tqdm>=4.62', 'dipy>=1.0'],
    scripts=glob('bin/*.py'),
    zip_safe=False
)