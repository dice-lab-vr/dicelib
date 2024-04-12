from dicelib.utils import get_version

from pathlib import Path

__version__ = get_version()

def get_include():
    include_dirs = []
    dir_path = Path(__file__).parent.resolve()
    include_dirs.append(str(dir_path))
    include_dirs.append(str(dir_path.joinpath('include')))
    return include_dirs
