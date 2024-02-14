from .lazytractogram import LazyTractogram
from pathlib import Path
from importlib import metadata

try:
    __version__ = metadata.version('dmri-dicelib')
except metadata.PackageNotFoundError:
    __version__ = 'not installed'

def get_include():
    include_dirs = []
    dir_path = Path(__file__).parent.resolve()
    include_dirs.append(str(dir_path))
    include_dirs.append(str(dir_path.joinpath('include')))
    return include_dirs
