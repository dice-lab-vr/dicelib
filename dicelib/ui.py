from datetime import datetime as _datetime
from sys import exit as _exit

# verbosity level of logging functions
__UI_VERBOSE_LEVEL__ = 2

# foreground colors
fBlack   = '\x1b[30m'
fRed     = '\x1b[31m'
fGreen   = '\x1b[32m'
fYellow  = '\x1b[33m'
fBlue    = '\x1b[34m'
fMagenta = '\x1b[35m'
fCyan    = '\x1b[36m'
fWhite   = '\x1b[37m'
fDefault = '\x1b[39m'

# foreground highlight colors (i.e. bold/bright)
hBlack   = '\x1b[30;1m'
hRed     = '\x1b[31;1m'
hGreen   = '\x1b[32;1m'
hYellow  = '\x1b[33;1m'
hBlue    = '\x1b[34;1m'
hMagenta = '\x1b[35;1m'
hCyan    = '\x1b[36;1m'
hWhite   = '\x1b[37;1m'

# background
bBlack   = '\x1b[40m'
bRed     = '\x1b[41m'
bGreen   = '\x1b[42m'
bYellow  = '\x1b[43m'
bBlue    = '\x1b[44m'
bMagenta = '\x1b[45m'
bCyan    = '\x1b[46m'
bWhite   = '\x1b[47m'
bDefault = '\x1b[49m'

# decorations
Reset     = '\x1b[0m'
Bold      = '\x1b[1m'
Underline = '\x1b[4m'
Reverse   = '\x1b[7m'


def set_verbose( verbose: int ):
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


def INFO( message: str ):
	"""Print a INFO message in blue.
	Only shown if __UI_VERBOSE_LEVEL__ == 2.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ == 2:
		print( fBlack+bCyan+"[ INFO ]"+fCyan+bDefault+" "+message+Reset )


def LOG( message: str ):
	"""Print a INFO message in green, reporting the time as well.
	Only shown if __UI_VERBOSE_LEVEL__ == 2.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ == 2:
		print( fBlack+bGreen+"[ "+_datetime.now().strftime("%H:%M:%S")+" ]"+fGreen+bDefault+" "+message+Reset )


def WARNING( message: str, stop: bool=False ):
	"""Print a WARNING message in yellow.
	Only shown if __UI_VERBOSE_LEVEL__ >= 1.

	Parameters
	----------
	message : string
		Message to display.
	stop : boolean
		If True, it stops the execution (default : False).
	"""
	if __UI_VERBOSE_LEVEL__ >= 1:
		print( fBlack+bYellow+"[ WARNING ]"+fYellow+bDefault+" "+message+Reset )
	if stop:
		_exit(1)


def ERROR( message: str, stop: bool=True ):
	"""Print an ERROR message in red.
	Only shown if __UI_VERBOSE_LEVEL__ >= 1.

	Parameters
	----------
	message : string
		Message to display.
	stop : boolean
		If True, it stops the execution (default : True).
	"""
	if __UI_VERBOSE_LEVEL__ >= 1:
		print( fBlack+bRed+"[ ERROR ]"+fRed+bDefault+" "+message+Reset )
	if stop:
		_exit(1)