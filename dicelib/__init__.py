from .info import __version__
from .lazytractogram import LazyTractogram


from pathlib import Path

def get_include():
    include_dirs = []
    dir_path = Path(__file__).parent.resolve()
    include_dirs.append(str(dir_path))
    include_dirs.append(str(dir_path.joinpath('include')))
    return include_dirs