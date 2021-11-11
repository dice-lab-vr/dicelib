from os.path import isfile, isdir
from datetime import datetime


def LOG( str ):
	print( "\033[7;36m[ %s ]\033[0;36m  %s \033[0m" % ( datetime.now().strftime("%H:%M:%S"), str ) )


def ERROR( str ):
	print( "\033[7;31m[ %s ]\033[0;31m  %s \033[0m" % ( datetime.now().strftime("%H:%M:%S"), str ) )


def WARNING( str ):
	print( "\033[7;33m[ %s ]\033[0;33m  %s \033[0m" % ( datetime.now().strftime("%H:%M:%S"), str ) )


def CHECK_FILE( filename, errorStr="" ):
	if isfile( filename ):
		return True
	if errorStr=="":
		ERROR( "Unable to locate file '%s'!" % filename )
	else:
		ERROR( errorStr  )
	return False


def CHECK_DIR( dirname, errorStr="" ):
	if isdir( dirname ):
		return True
	if errorStr=="":
		ERROR( "Unable to locate folder '%s'!" % dirname )
	else:
		ERROR( errorStr  )
	return False
