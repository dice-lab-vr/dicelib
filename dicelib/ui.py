from datetime import datetime as _datetime
from sys import exit as _exit


# verbosity level of logging functions
__UI_VERBOSE_LEVEL__ = 2


def set_verbose( verbose ):
	"""Set the verbosity of logging functions.

	Parameters
	----------
	verbose : int
		0=show nothing, 1=show only warnings/errors, 2=show all
	"""
	global __UI_VERBOSE_LEVEL__
	if type(verbose) != int or verbose not in [0,1,2]:
		raise TypeError( '"verbose" must be either 0, 1 or 2' )
	__UI_VERBOSE_LEVEL__ = verbose


def INFO( str ):
	global __UI_VERBOSE_LEVEL__
	if __UI_VERBOSE_LEVEL__ == 2:
		print( "\033[7;36m[ INFO ]\033[0;36m %s \033[0m" % str )


def LOG( str ):
	global __UI_VERBOSE_LEVEL__
	if __UI_VERBOSE_LEVEL__ == 2:
		print( "\033[7;32m[ %s ]\033[0;32m %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )


def WARNING( str, stop=False ):
	global __UI_VERBOSE_LEVEL__
	if __UI_VERBOSE_LEVEL__ >= 1:
		print( "\033[7;33m[ WARNING ]\033[0;33m %s \033[0m" % str )
	if stop:
		_exit(1)


def ERROR( str, stop=True ):
	global __UI_VERBOSE_LEVEL__
	if __UI_VERBOSE_LEVEL__ >= 1:
		print( "\033[7;31m[ ERROR ]\033[0;31m %s \033[0m" % str )
	if stop:
		_exit(1)