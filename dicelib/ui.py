from datetime import datetime as _datetime
from sys import exit as _exit


def LOG( str ):
	print( "\033[7;36m[ %s ]\033[0;36m  %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )


def WARNING( str, stop=False ):
	print( "\033[7;33m[ %s ]\033[0;33m  %s \033[0m" % ( _datetime.now().strftime("%H:%M:%S"), str ) )
	if stop:
		_exit(1)


def ERROR( str, stop=True ):
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
