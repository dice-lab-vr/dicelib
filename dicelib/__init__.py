__all__ = ['ui', 'clustering']
from . import ui
from . import clustering

from pkg_resources import get_distribution as _get_distribution
__version__ = _get_distribution('dicelib').version
