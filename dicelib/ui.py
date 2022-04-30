from datetime import datetime as _datetime
import sys as _sys
from argparse import ArgumentParser as _ArgumentParser, ArgumentDefaultsHelpFormatter as _ArgumentDefaultsHelpFormatter

# verbosity level of logging functions
__UI_VERBOSE_LEVEL__ = 3

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
	"""Set the verbosity of all functions.

	Parameters
	----------
	verbose : int
        3 = show everything
		2 = show messages but no progress bars
        1 = show only warnings/errors
        0 = hide everything
	"""
	global __UI_VERBOSE_LEVEL__
	if type(verbose) != int or verbose not in [0,1,2,3]:
		raise TypeError( '"verbose" must be either 0, 1, 2 or 3' )
	__UI_VERBOSE_LEVEL__ = verbose


def get_verbose():
    return __UI_VERBOSE_LEVEL__


def PRINT( *args, **kwargs ):
    if __UI_VERBOSE_LEVEL__ >= 2:
        print( *args, **kwargs )

def INFO( message: str ):
	"""Print a INFO message in blue.
	Only shown if __UI_VERBOSE_LEVEL__ >= 2.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ >= 2:
		print( fBlack+bCyan+"[ INFO ]"+fCyan+bDefault+" "+message+Reset )


def LOG( message: str ):
	"""Print a INFO message in green, reporting the time as well.
	Only shown if __UI_VERBOSE_LEVEL__ >= 2.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ >= 2:
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
		_sys.exit()


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
		_sys.exit()


class ColoredArgParser( _ArgumentParser ):
	"""Modification of 'argparse.ArgumentParser' to allow colored output.
	"""
	class _ColoredFormatter( _ArgumentDefaultsHelpFormatter ):
		COLOR = fMagenta

		def start_section(self, heading):
			super().start_section( Underline+heading.capitalize()+Reset )

		def _format_action(self, action):
			# determine the required width and the entry label
			help_position = min(self._action_max_length + 2, self._max_help_position)
			help_width = max(self._width - help_position, 11)
			action_width = help_position - self._current_indent - 2
			action_header = self._format_action_invocation(action)

			# no help; start on same line and add a final newline
			if not action.help:
				tup = self._current_indent, '', action_header
				action_header = '%*s%s\n' % tup

			# short action name; start on the same line and pad two spaces
			elif len(action_header) <= action_width:
				tup = self._current_indent, '', action_width, action_header
				action_header = '%*s%-*s  ' % tup
				indent_first = 0

			# long action name; start on the next line
			else:
				tup = self._current_indent, '', action_header
				action_header = '%*s%s\n' % tup
				indent_first = help_position

			# collect the pieces of the action help
			parts = [ action_header ]

			# add color codes
			for i in range(len(parts)):
				tmp = parts[i].split(',')
				parts[i] = ','.join( [self.COLOR+s+Reset for s in tmp] )

			# if there was help for the action, add lines of help text
			if action.help and action.help.strip():
				help_text = self._expand_help(action)
				if help_text:
					help_lines = self._split_lines(help_text, help_width)
					parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
					for line in help_lines[1:]:
						parts.append('%*s%s\n' % (help_position, '', line))

			# or add a newline if the description doesn't end with one
			elif not action_header.endswith('\n'):
				parts.append('\n')

			# if there are any sub-actions, add their help as well
			for subaction in self._iter_indented_subactions(action):
				parts.append(self._format_action(subaction))

			# return a single string
			return self._join_parts(parts)

		def _format_usage(self, usage, actions, groups, prefix):
			return super()._format_usage( usage, actions, groups, prefix='USAGE:  '+self.COLOR ) +Reset


	def __init__( self, *args, **kwargs ):
		super().__init__( formatter_class=self._ColoredFormatter, *args, **kwargs )


	def parse_known_args(self, args=None, namespace=None):
		if args is None:
			args = _sys.argv[1:]
		else:
			args = list(args)
		if len(args)==0:
			self.print_help()
			_sys.exit()
		return super().parse_known_args(args, namespace)


	def error( self, message ):
		self.print_usage()
		ERROR( message )