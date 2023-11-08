from datetime import datetime as _datetime
import sys as _sys
from argparse import ArgumentParser as _ArgumentParser, ArgumentDefaultsHelpFormatter as _ArgumentDefaultsHelpFormatter
import itertools
import numpy as np
from threading import Thread
from time import time, sleep
from shutil import get_terminal_size

def _in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # IPython terminal
        else:
            return False # Other terminal
    except NameError:
        pass # Python interpreter

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


# verbosity level of logging functions
__UI_VERBOSE_LEVEL__ = 4


def set_verbose( verbose: int ):
	"""Set the verbosity of all functions.

	Parameters
	----------
	verbose : int
        4 = show everything
		3 = show all messages but no progress
		2 = show warnings/errors and progress
        1 = show warnings/errors but no progress
        0 = hide everything
	"""
	global __UI_VERBOSE_LEVEL__
	if type(verbose) != int or verbose not in [0,1,2,3,4]:
		raise TypeError( '"verbose" must be either 0, 1, 2, 3 or 4' )
	__UI_VERBOSE_LEVEL__ = verbose


def get_verbose():
    return __UI_VERBOSE_LEVEL__


def PRINT( *args, **kwargs ):
    if __UI_VERBOSE_LEVEL__ >= 3:
        print( *args, **kwargs )

def INFO( message: str ):
	"""Print a INFO message in blue.
	Only shown if __UI_VERBOSE_LEVEL__ >= 3.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ >= 3:
		print( fBlack+bCyan+"[ INFO ]"+fCyan+bDefault+" "+message+Reset )


def LOG( message: str ):
	"""Print a INFO message in green, reporting the time as well.
	Only shown if __UI_VERBOSE_LEVEL__ >= 3.

	Parameters
	----------
	message : string
		Message to display.
	"""
	if __UI_VERBOSE_LEVEL__ >= 3:
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

class ProgressBar:
    """Class that provides a progress bar during long-running processes.
    
    It can be used either as a indeterminate or determinate progress bar.
    Determinate progress bar supports multithread progress tracking.
    It can be used as a context manager.

    Parameters
    ----------
    total : int or None
        Total number of steps. If None, an indeterminate progress bar is used (default is None).
    ncols : int
        Number of columns of the progress bar in the terminal (default is 58).
    refresh : float
        Refresh rate of the progress bar in seconds (default is 0.05).
    eta_refresh : float
        Refresh rate of the estimated time of arrival in seconds (default 1).
    multithread_progress : (nthreads,) np.ndarray or None
        Array that contains the progress of each thread. If None, the progress
		is tracked as singlethreaded (default is None).
    disable : bool
        Whether to disable the progress bar (default is False).
	
	Examples
	--------
	Indeterminate progress bar.

	>>> with ProgressBar():
	...     my_long_running_function()

	Determinate singlethread progress bar.

	>>> with ProgressBar(total=100) as pbar:
	...     for i in range(100):
	...         # some operations
	...         pbar.update()

	Determinate multithread progress bar.

	>>> progress = np.zeros(4)
	>>> with ProgressBar(total=400, multithread_progress=progress) as pbar:
	...     my_multithread_function(progress, thread_id)

	...     # in each thread
	...     for i in range(100):
	...         # some operations
	...         progress[thread_id] += 1
    """
    
    def __init__(self, total=None, ncols=None, refresh=0.05, eta_refresh=1, multithread_progress=None, hide_on_exit=True, disable=False):
        self.total = total
        self.ncols = int(get_terminal_size().columns // 2) if ncols is None else ncols
        self.refresh = refresh
        self.eta_refresh = eta_refresh
        self.multithread_progress = multithread_progress
        self.hide_on_exit = hide_on_exit
        self.disable = disable

        self._graphics = {
            'clear_line': '\x1b[2K' if not _in_notebook() else f"\r{' '*get_terminal_size().columns*2}",
            'reset': '\x1b[0m',
            'black': '\x1b[30m',
            'green': '\x1b[32m',
            'magenta': '\x1b[35m',
            'cyan': '\x1b[36m',
            'bright_black': '\x1b[90m'
        }

        self._done = False

        if self.total is None:
            bar_length = int(self.ncols // 2)
            self._steps = []
            for i in range(self.ncols - bar_length + 1):
                self._steps.append(f"{self._graphics['bright_black']}{'━' * i}{self._graphics['magenta']}{'━' * bar_length}{self._graphics['bright_black']}{'━' * (self.ncols - bar_length - i)}")
            for i in range(bar_length - 1):
                self._steps.append(f"{self._graphics['magenta']}{'━' * (i + 1)}{self._graphics['bright_black']}{'━' * (self.ncols - bar_length)}{self._graphics['magenta']}{'━' * (bar_length - i - 1)}")
        else:
            self._eta = '<eta --m --s>'
            self._start_time = 0
            self._last_time = 0
            self._progress = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _update_eta(self):
        self._last_time = time()
        if self.multithread_progress is not None:
            self._progress = np.sum(self.multithread_progress)
        eta = (time() - self._start_time) * (self.total - self._progress) / self._progress if self._progress > 0 else 0
        self._eta = f'<eta {int(eta // 60):02d}m {int(eta % 60):02d}s>'

    def _animate(self):
        if self.total is None:
            for step in itertools.cycle(self._steps):
                if self._done:
                    break
                print(f"\r   {step}{self._graphics['reset']}", end='', flush=True)
                sleep(self.refresh)
        else:
            while True:
                if self._done:
                    break
                if time() - self._last_time > self.eta_refresh:
                    self._update_eta()
                if self.multithread_progress is not None:
                    self._progress = np.sum(self.multithread_progress)
                print(f"\r   {self._graphics['magenta']}{'━' * int(self.ncols * self._progress / self.total)}{self._graphics['bright_black']}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {self._graphics['green']}{100 * self._progress / self.total:.1f}% {self._graphics['cyan']}{self._eta}{self._graphics['reset']}", end='', flush=True)
                sleep(self.refresh)

    def start(self):
        if not self.disable:
            if self.total is not None:
                self._start_time = time()
                self._last_time = self._start_time
            Thread(target=self._animate, daemon=True).start()

    def stop(self):
        self._done = True
        if not self.disable:
            print(self._graphics['clear_line'], end='\r', flush=True)
            if not self.hide_on_exit:
                if self.total is None:
                    print(f"\r   {self._graphics['green']}{'━' * self.ncols} 100.0%{self._graphics['reset']}")
                else:
                    if self.multithread_progress is not None:
                        self._progress = np.sum(self.multithread_progress)
                    print(f"\r   {self._graphics['green']}{'━' * int(self.ncols * self._progress / self.total)}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {100 * self._progress / self.total:.1f}%{self._graphics['reset']}")

    def update(self):
        self._progress += 1
