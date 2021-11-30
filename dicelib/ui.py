from datetime import datetime as _datetime
from sys import exit as _exit

__UI_IS_VERBOSE__ = True

def set_verbose( verbose ):
	if type(verbose) != bool:
		raise TypeError( '"verbose" must be boolean' )
	__UI_IS_VERBOSE__ = verbose


def LOG( str ):
	if __UI_IS_VERBOSE__:
		print( "\033[7;36m[ %s ]\033[0;36m  %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )


def WARNING( str, stop=False ):
	if __UI_IS_VERBOSE__:
		print( "\033[7;33m[ %s ]\033[0;33m  %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )
	if stop:
		_exit(1)


def ERROR( str, stop=True ):
	if __UI_IS_VERBOSE__:
		print( "\033[7;31m[ %s ]\033[0;31m  %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )
	if stop:
		_exit(1)


def CHECK_FILE( filename, raise_error=False ):
	from os.path import isfile
	if isfile( filename ):
		return True
	if raise_error:
		raise FileNotFoundError( f'Unable to locate file "{filename}"' )
	return False


def CHECK_DIR( dirname, raise_error=False ):
	from os.path import isdir
	if isdir( dirname ):
		return True
	if raise_error:
		raise FileNotFoundError( f'Unable to locate folder "{dirname}"' )
	return False
