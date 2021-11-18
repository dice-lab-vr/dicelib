# __all__ = ['ui', 'tractogram']
# from . import ui
# from . import tractogram

from pkg_resources import get_distribution as _get_distribution
__version__ = _get_distribution('dicelib').version
