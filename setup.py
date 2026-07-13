from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from numpy import get_include
from shutil import rmtree
import os, sys, subprocess

package_name = 'dicelib'


def get_extensions(extra_compile_args, extra_link_args):
    image = Extension(
        name=f'{package_name}.image',
        sources=[f'{package_name}/image.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native'],
        language='c++'
    )
    streamline = Extension(
        name=f'{package_name}.streamline',
        sources=[f'{package_name}/streamline.pyx'],
        include_dirs=[get_include(), f'{package_name}/include'],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native'],
        language='c++'
    )
    tractogram = Extension(
        name=f'{package_name}.tractogram',
        sources=[f'{package_name}/tractogram.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native']+extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    )
    clustering = Extension(
        name=f'{package_name}.clustering',
        sources=[f'{package_name}/clustering.pyx'],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native'],
        language='c++',
    )
    connectivity = Extension(
        name=f'{package_name}.connectivity',
        sources=['dicelib/connectivity.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native'],
        language='c++',
    )
    tsf = Extension(
        name=f'{package_name}.tsf',
        sources=[f'{package_name}/tsf.pyx'],
        include_dirs=[get_include()],
        extra_compile_args=['-w', '-std=c++11', '-g0', '-O3', '-march=native'],
        language='c++',
    )
    return [image, streamline, tractogram, clustering, connectivity, tsf]


class CustomBuildExtCommand(build_ext):
    """build_ext command to use when numpy headers are needed"""

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include
        from multiprocessing import cpu_count

        # Add everything requires for build
        self.swig_opts = None
        self.include_dirs = [get_include()]
        self.distribution.ext_modules[:] = cythonize(self.distribution.ext_modules, build_dir='build')

        # if not specified via '-j N' option, set compilation using max number of cores
        if self.parallel is None:
            self.parallel = cpu_count()
        print( f'Parallel compilation using {self.parallel} threads' )

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
        rmtree('./build', ignore_errors=True)

# Check for OPENMP support
OPENMP_compile_args = []
OPENMP_link_args = []
if sys.platform.startswith("win"):
    compiler = "cl"
    flags = ["/openmp", "/Fe:NUL", "/TC", "-"]
    OPENMP_compile_args = ["/openmp"]
    OPENMP_link_args = []
else:
    compiler = os.environ.get("CC", "gcc")
    flags = ["-fopenmp", "-o", "/dev/null", "-x", "c", "-"]
    OPENMP_compile_args = ["-fopenmp"]
    OPENMP_link_args = ["-fopenmp"]

try:
    result = subprocess.run(
        [compiler] + flags,
        input="#include <omp.h>\nint main(void) { return omp_get_num_threads(); }",
        text=True, capture_output=True, timeout=5
    )
    if result.returncode != 0:
        OPENMP_compile_args = []
        OPENMP_link_args = []
except Exception:
    OPENMP_compile_args = []
    OPENMP_link_args = []

setup(
    cmdclass={
        'build_ext': CustomBuildExtCommand,
        'clean': CleanCommand
    },
    ext_modules=get_extensions(OPENMP_compile_args, OPENMP_link_args)
)