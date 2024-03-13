from dicelib import __version__

import numpy as np

import argparse
from datetime import datetime as _datetime
import itertools
import logging
import re as _re
from shutil import get_terminal_size
import sys
import textwrap
from threading import Thread
from time import sleep, time
from typing import Literal, NoReturn, TypeAlias

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

# ASCII art
ascii_art = f'''\
    ██████╗ ██╗ ██████╗███████╗██╗     ██╗██████╗
    ██╔══██╗██║██╔════╝██╔════╝██║     ██║██╔══██╗
    ██║  ██║██║██║     █████╗  ██║     ██║██████╔╝
    ██║  ██║██║██║     ██╔══╝  ██║     ██║██╔══██╗
    ██████╔╝██║╚██████╗███████╗███████╗██║██████╔╝
    ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚══════╝╚═╝╚═════╝  [v{__version__}]
'''

# ANSI escape codes
esc = '\x1b['
reset = f'{esc}0m'
default = f'{esc}39m'
fg = f'{esc}38;2;'
bg = f'{esc}48;2;'

# text formatting and effects
text_underline = f'{esc}4m'
text_blink = f'{esc}5m'
clear_line = f'{esc}2K'
# terminal_size = ' ' * get_terminal_size().columns * 2
# clear_line = f'{esc}2K' if not _in_notebook() else f'\r{terminal_size}'

# base colors (rgb)
black = '0;0;0'
white = '255;255;255'
red = '255;0;0'
green = '0;255;0'
blue = '0;0;255'
yellow = '255;255;0'
magenta = '255;0;255'
cyan = '0;255;255'
orange = '255;128;0'
pink = '255;0;128'
light_green = '128;255;128'
light_blue = '128;128;255'

# foreground colors
fg_black = f'{fg}{black}m'
fg_red = f'{fg}{red}m'
fg_green = f'{fg}{green}m'
fg_blue = f'{fg}{blue}m'
fg_yellow = f'{fg}{yellow}m'
fg_magenta = f'{fg}{magenta}m'
fg_cyan = f'{fg}{cyan}m'
fg_orange = f'{fg}{orange}m'
fg_pink = f'{fg}{pink}m'
fg_light_green = f'{fg}{light_green}m'
fg_light_blue = f'{fg}{light_blue}m'

# background colors
bg_red = f'{bg}{red}m'
bg_green = f'{bg}{green}m'
bg_yellow = f'{bg}{yellow}m'
bg_cyan = f'{bg}{cyan}m'

# Logger
Mode: TypeAlias = Literal['console', 'file']
class LoggerFormatter(logging.Formatter):
    def __init__(self, mode: Mode) -> NoReturn:
        self.levelname_len = 10
        self.message_len = 60
        self.levelname_sep = '-'
        self.message_sep = ' '
        asctime = '{asctime}'
        levelname = '{levelname}'
        message = '{message}'
        module = '{module}'
        lineno = '{lineno}'

        if mode == 'console':
            self.message_indent = 13
            self.formats = {
                logging.DEBUG: f'{fg_green}-{levelname}>{reset}  {message}  {fg_green}-{module}:{lineno} ({asctime})-{reset}',
                logging.INFO: f'{bg_cyan}{fg_black}-{levelname}>{reset}  {message}',
                logging.WARNING: f'{bg_yellow}{fg_black}-{levelname}>{reset}  {message}  {fg_yellow}-{module}:{lineno}-{reset}',
                logging.ERROR: f'{bg_red}{fg_black}-{levelname}>{reset}  {message}  {fg_red}-{module}:{lineno}-{reset}',
                logging.CRITICAL: f'{bg_red}{fg_black}-{levelname}>{reset}  {message}  {fg_red}-{module}:{lineno}-{reset}'
            }
            self.format_levelname = lambda text: f'{text}'.ljust(self.levelname_len - 1, self.levelname_sep)
        elif mode == 'file':
            self.message_indent = 14
            self.formats = {
                logging.DEBUG: f'[{levelname}]  {message}  {module}:{lineno} ({asctime})',
                logging.INFO: f'[{levelname}]  {message}',
                logging.WARNING: f'[{levelname}]  {message}  {module}:{lineno}',
                logging.ERROR: f'[{levelname}]  {message}  {module}:{lineno}',
                logging.CRITICAL: f'[{levelname}]  {message}  {module}:{lineno}'
            }
            self.format_levelname = lambda text: f'{text}'.center(self.levelname_len, self.levelname_sep)
        else:
            raise ValueError('"mode" must be "console" or "file"')
        super().__init__(fmt=self.formats[logging.DEBUG], datefmt='%H:%M:%S', style='{')
    
    def format(self, record):
        super().__init__(fmt=self.formats[record.levelno], datefmt='%H:%M:%S', style='{')
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        record.levelname = self.format_levelname(logging.getLevelName(record.levelno))

        if len(record.message) > self.message_len:
            rows = textwrap.wrap(textwrap.dedent(record.message), width=self.message_len)
            s = f'{rows[0].ljust(self.message_len, self.message_sep)}\n'
            for row in rows[1:-1]:
                s += f'{self.message_sep*self.message_indent}{row.ljust(self.message_len, self.message_sep)}\n'
            s += f'{self.message_sep*self.message_indent}{rows[-1].ljust(self.message_len, self.message_sep)}'
            record.message = s
        else:
            s = textwrap.dedent(record.message.strip())
            s = s.ljust(self.message_len, self.message_sep)
            record.message = s
        
        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s

# Argument parser
class ArgumentParserFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    def _format_actions_usage(self, actions, groups):
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f'empty group {group}')

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                group_action_count = len(group._group_actions)
                end = start + group_action_count
                if actions[start:end] == group._group_actions:

                    suppressed_actions_count = 0
                    for action in group._group_actions:
                        group_actions.add(action)
                        if action.help is argparse.SUPPRESS:
                            suppressed_actions_count += 1

                    exposed_actions_count = group_action_count - suppressed_actions_count

                    if not group.required:
                        if start in inserts:
                            inserts[start] += ' ['
                        else:
                            inserts[start] = '['
                        if end in inserts:
                            inserts[end] += ']'
                        else:
                            inserts[end] = ']'
                    elif exposed_actions_count > 1:
                        if start in inserts:
                            inserts[start] += ' ('
                        else:
                            inserts[start] = '('
                        if end in inserts:
                            inserts[end] += ')'
                        else:
                            inserts[end] = ')'
                    for i in range(start + 1, end):
                        inserts[i] = '|'

        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is argparse.SUPPRESS:
                parts.append(None)
                if inserts.get(i) == '|':
                    inserts.pop(i)
                elif inserts.get(i + 1) == '|':
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # NOTE: add colors
        # positional arguments
        # if action.choices -> opt = {c1,c2}
        #    opt
        # ?  [opt]
        # *  [opt ...]
        # +  opt [opt ...]

        # optional arguments
        # if action.choices and no metavar -> var = {c1,c2}
        #    [opt var]
        # ?  [opt [var]]
        # *  [opt [var ...]]
        # +  [opt var [var ...]]
        # r  opt var
        # r? opt [var]
        # r* opt [var ...]
        # r+ opt var [var ...]
        for i, part in enumerate(parts):
            part = part.strip()
            if part.startswith('['):
                if part.endswith(']]'):
                    spaces = part.count(' ')
                    j = part.find(' ')
                    if spaces == 1:
                        # '[opt [{choices}]]' if action.choices else '[opt [var]]'
                        parts[i] = f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:-3]}{reset}{part[-3:]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-2]}{reset}{part[-2:]}'
                    elif spaces == 2:
                        # '[opt [{choices} ...]]' if action.choices else '[opt [var ...]]'
                        jj = part[j + 1:].find(' ') + j + 1
                        parts[i] = f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:jj - 1]}{reset}{part[jj - 1]} {fg_light_green}{part[jj + 1:-2]}{reset}{part[-2:]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj]}{reset} {fg_light_green}{part[jj + 1:-2]}{reset}{part[-2:]}'
                    else:
                        # '[opt {choices} [{choices} ...]]' if action.choices else '[opt var [var ...]]'
                        jj = part[j + 1:].find(' ') + j + 1
                        jjj = part[jj + 1:].find(' ') + jj + 1
                        parts[i] = f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj - 1]}{reset}{part[jj - 1]} {part[jj + 1:jj + 3]}{fg_green}{part[jj + 3:jjj - 1]}{reset}{part[jjj - 1]} {fg_light_green}{part[jjj + 1:-2]}{reset}{part[-2:]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:j]}{reset} {fg_green}{part[j + 1:jj]}{reset} {part[jj + 1]}{fg_green}{part[jj + 2:jjj]}{reset} {fg_light_green}{part[jjj + 1:-2]}{reset}{part[-2:]}'
                else:
                    if ' ' in part:
                        j = part.find(' ')
                        if actions[i].nargs == '*':
                            # '[{choices} ...]' if action.choices else '[opt ...]'
                            parts[i] = f'{part[:2]}{fg_blue}{part[2:j - 1]}{reset}{part[j - 1]} {fg_light_blue}{part[j + 1:-1]}{reset}{part[-1]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:j]}{reset} {fg_light_blue}{part[j + 1:-1]}{reset}{part[-1]}'
                        else:
                            # '[opt {choices}]' if action.choices else '[opt var]'
                            parts[i] = f'{part[0]}{fg_blue}{part[1:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-2]}{reset}{part[-2:]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:j]}{reset} {fg_green}{part[j + 1:-1]}{reset}{part[-1]}'
                    else:
                        # '[{choices}]' if action.choices else '[opt]'
                        parts[i] = f'{part[:2]}{fg_blue}{part[2:-2]}{reset}{part[-2:]}' if actions[i].choices else f'{part[0]}{fg_blue}{part[1:-1]}{reset}{part[-1]}'
            elif part.endswith(']'):
                spaces = part.count(' ')
                j = part.find(' ')
                if spaces == 1:
                    # 'opt [{choices}]' if action.choices else 'opt [var]'
                    parts[i] = f'{fg_blue}{part[:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:-2]}{reset}{part[-2:]}' if actions[i].choices else f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-1]}{reset}{part[-1]}'
                elif spaces == 2:
                    # 'opt [opt ...]' if action.nargs is + else 'opt [var ...]'
                    jj = part[j + 1:].find(' ') + j + 1
                    if actions[i].nargs == '+':
                        # '{choices} [{choices} ...]' if action.choices else 'opt [opt ...]'
                        parts[i] = f'{part[0]}{fg_blue}{part[1:j - 1]}{reset}{part[j - 1]} {part[j + 1:j + 3]}{fg_blue}{part[j + 3:jj - 1]}{reset}{part[jj - 1]} {fg_light_blue}{part[jj + 1:-1]}{reset}{part[-1]}' if actions[i].choices else f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj]}{reset} {fg_light_green}{part[jj + 1:-1]}{reset}{part[-1]}'
                    else:
                        # 'opt [{choices} ...]' if action.choices else 'opt [var ...]'
                        parts[i] = f'{fg_blue}{part[:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:jj - 1]}{reset}{part[jj - 1]} {fg_light_green}{part[jj + 1:-1]}{reset}{part[-1]}' if actions[i].choices else f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj]}{reset} {fg_light_green}{part[jj + 1:-1]}{reset}{part[-1]}'
                else:
                    # 'opt {choices} [{choices} ...]' if action.choices else 'opt var [var ...]'
                    jj = part[j + 1:].find(' ') + j + 1
                    jjj = part[jj + 1:].find(' ') + jj + 1
                    parts[i] = f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj - 1]}{reset}{part[jj - 1]} {part[jj + 1:jj + 3]}{fg_green}{part[jj + 3:jjj - 1]}{reset}{part[jjj - 1]} {fg_light_green}{part[jjj + 1:-1]}{reset}{part[-1]}' if actions[i].choices else f'{fg_blue}{part[:j]}{reset} {fg_green}{part[j + 1:jj]}{reset} {part[jj + 1]}{fg_green}{part[jj + 2:jjj]}{reset} {fg_light_green}{part[jjj + 1:-1]}{reset}{part[-1]}'
            else:
                if ' ' in part:
                    # 'opt {choices}' if action.choices else 'opt var'
                    j = part.find(' ')
                    parts[i] =f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-1]}{reset}{part[-1]}' if actions[i].choices else f'{fg_blue}{part[:j]}{reset} {fg_green}{part[j + 1:]}{reset}'
                else:
                    # '{choices}' if action.choices else 'opt'
                    parts[i] = f'{part[0]}{fg_blue}{part[1:-1]}{reset}{part[-1]}' if actions[i].choices else f'{fg_blue}{part}{reset}'

        # join all the action items with spaces
        text = ' '.join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open = r'[\[(]'
        close = r'[\])]'
        text = _re.sub(r'(%s) ' % open, r'\1', text)
        text = _re.sub(r' (%s)' % close, r'\1', text)
        text = _re.sub(r'%s *%s' % (open, close), r'', text)
        text = text.strip()

        # return the text
        return text
        
    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
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
        parts = [action_header]

        # NOTE: add colors
        # positional arguments
        # if action.choices -> opt = {c1,c2}
        #    opt

        # optional arguments (possibly multiple arguments separated by ', ')
        # if action.choices and no metavar -> var = {c1,c2}
        #    opt var
        # ?  opt [var]
        # *  opt [var ...]
        # +  opt var [var ...]
        for i, part in enumerate(parts):
            part = part.strip()
            if ',' in part:
                k = 0
                colored_text = ''
                while k <= len(part):
                    m = part[k:].find(', ')
                    n = (m + k) if m != -1 else len(part)
                    tmp_text = part[k:n]
                    if tmp_text.endswith(']'):
                        spaces = tmp_text.count(' ')
                        j = tmp_text.find(' ')
                        if spaces == 1:
                            # 'opt [{choices}]' if action.choices else 'opt [var]'
                            colored_text += f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1:j + 3]}{fg_green}{tmp_text[j + 3:-2]}{reset}{tmp_text[-2:]}' if action.choices else f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1]}{fg_green}{tmp_text[j + 2:-1]}{reset}{tmp_text[-1]}'
                        elif spaces == 2:
                            # 'opt [{choices} ...]' if action.choices else 'opt [var ...]'
                            jj = tmp_text[j + 1:].find(' ') + j + 1
                            colored_text += f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1:j + 3]}{fg_green}{tmp_text[j + 3:jj - 1]}{reset}{tmp_text[jj - 1]} {fg_light_green}{tmp_text[jj + 1:-1]}{reset}{tmp_text[-1]}' if action.choices else f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1]}{fg_green}{tmp_text[j + 2:jj]}{reset} {fg_light_green}{tmp_text[jj + 1:-1]}{reset}{tmp_text[-1]}'
                        else:
                            # 'opt {choices} [{choices} ...]' if action.choices else 'opt var [var ...]'
                            jj = tmp_text[j + 1:].find(' ') + j + 1
                            jjj = tmp_text[jj + 1:].find(' ') + jj + 1
                            colored_text += f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1]}{fg_green}{tmp_text[j + 2:jj - 1]}{reset}{tmp_text[jj - 1]} {tmp_text[jj + 1:jj + 3]}{fg_green}{tmp_text[jj + 3:jjj - 1]}{reset}{tmp_text[jjj - 1]} {fg_light_green}{tmp_text[jjj + 1:-1]}{reset}{tmp_text[-1]}' if action.choices else f'{fg_blue}{tmp_text[:j]}{reset} {fg_green}{tmp_text[j + 1:jj]}{reset} {tmp_text[jj + 1]}{fg_green}{tmp_text[jj + 2:jjj]}{reset} {fg_light_green}{tmp_text[jjj + 1:-1]}{reset}{tmp_text[-1]}'
                    else:
                        if ' ' in tmp_text:
                            # 'opt {choices}' if action.choices else 'opt var'
                            j = tmp_text.find(' ')
                            colored_text += f'{fg_blue}{tmp_text[:j]}{reset} {tmp_text[j + 1]}{fg_green}{tmp_text[j + 2:-1]}{reset}{tmp_text[-1]}' if action.choices else f'{fg_blue}{tmp_text[:j]}{reset} {fg_green}{tmp_text[j + 1:]}{reset}'
                        else:
                            # '{choices}' if action.choices else 'opt'
                            colored_text += f'{tmp_text[0]}{fg_blue}{tmp_text[1:-1]}{reset}{tmp_text[-1]}' if action.choices else f'{fg_blue}{tmp_text}{reset}'
                    if n != len(part):
                        colored_text += ', '
                    k = n + 2
                parts[i] = parts[i].replace(part, colored_text)
            elif part.endswith(']'):
                spaces = part.count(' ')
                j = part.find(' ')
                if spaces == 1:
                    # 'opt [{choices}]' if action.choices else 'opt [var]'
                    parts[i] = parts[i].replace(part, f'{fg_blue}{part[:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:-2]}{reset}{part[-2:]}' if action.choices else f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-1]}{reset}{part[-1]}')
                elif spaces == 2:
                    # 'opt [{choices} ...]' if action.choices else 'opt [var ...]'
                    jj = part[j + 1:].find(' ') + j + 1
                    parts[i] = parts[i].replace(part, f'{fg_blue}{part[:j]}{reset} {part[j + 1:j + 3]}{fg_green}{part[j + 3:jj - 1]}{reset}{part[jj - 1]} {fg_light_green}{part[jj + 1:-1]}{reset}{part[-1]}' if action.choices else f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj]}{reset} {fg_light_green}{part[jj + 1:-1]}{reset}{part[-1]}')
                else:
                    # 'opt {choices} [{choices} ...]' if action.choices else 'opt var [var ...]'
                    jj = part[j + 1:].find(' ') + j + 1
                    jjj = part[jj + 1:].find(' ') + jj + 1
                    parts[i] = parts[i].replace(part, f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:jj - 1]}{reset}{part[jj - 1]} {part[jj + 1:jj + 3]}{fg_green}{part[jj + 3:jjj - 1]}{reset}{part[jjj - 1]} {fg_light_green}{part[jjj + 1:-1]}{reset}{part[-1]}' if action.choices else f'{fg_blue}{part[:j]}{reset} {fg_green}{part[j + 1:jj]}{reset} {part[jj + 1]}{fg_green}{part[jj + 2:jjj]}{reset} {fg_light_green}{part[jjj + 1:-1]}{reset}{part[-1]}')
            else:
                if ' ' in part:
                    # 'opt {choices}' if action.choices else 'opt var'
                    j = part.find(' ')
                    parts[i] = parts[i].replace(part, f'{fg_blue}{part[:j]}{reset} {part[j + 1]}{fg_green}{part[j + 2:-1]}{reset}{part[-1]}' if action.choices else f'{fg_blue}{part[:j]}{reset} {fg_green}{part[j + 1:]}{reset}')
                else:
                    # '{choices}' if action.choices else 'opt'
                    parts[i] = parts[i].replace(part, f'{part[0]}{fg_blue}{part[1:-1]}{reset}{part[-1]}' if action.choices else f'{fg_blue}{part}{reset}')

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

    class _Section(argparse.HelpFormatter._Section):
        def format_help(self):
            # format the indented section
            if self.parent is not None:
                self.formatter._indent()
            join = self.formatter._join_parts
            item_help = join([func(*args) for func, args in self.items])
            if self.parent is not None:
                self.formatter._dedent()

            # return nothing if the section was empty
            if not item_help:
                return ''

            # add the heading if the section was non-empty
            if self.heading is not argparse.SUPPRESS and self.heading is not None:
                current_indent = self.formatter._current_indent
                heading = '%*s%s:\n' % (current_indent, '', f'{fg_orange}{text_underline}{self.heading.upper()}{reset}') # NOTE: add format and color
            else:
                heading = ''

            # join the section-initial newline, the heading and the help
            return join(['\n', heading, item_help, '\n'])
        
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage' # NOTE: change default prefix

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + 2 + len(usage) > text_width: # NOTE: add 2 to account for ': '

                # break usage into wrappable parts
                part_regexp = (
                    r'\(.*?\)+(?=\s|$)|'
                    r'\[.*?\]+(?=\s|$)|'
                    r'\S+'
                )
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = _re.findall(part_regexp, opt_usage)
                pos_parts = _re.findall(part_regexp, pos_usage)
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + 2 + len(prog) <= 0.75 * text_width: # NOTE: add 2 to account for ': '
                    indent = ' ' * (len(prefix) + 2 + len(prog) + 1)
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix) + 2 # NOTE: add 2 to account for ': '
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'PREFIX: '
        return f'{fg_orange}{text_underline}{prefix.upper()}{reset}: {usage}\n\n' # NOTE: add format and color

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=[],
                 formatter_class=ArgumentParserFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=False,
                 allow_abbrev=True,
                 exit_on_error=True):
        super().__init__(prog,
                         usage,
                         textwrap.dedent(description) if description is not None else None, # NOTE: dedent description
                         epilog,
                         parents,
                         formatter_class,
                         prefix_chars,
                         fromfile_prefix_chars,
                         argument_default,
                         conflict_handler,
                         add_help,
                         allow_abbrev,
                         exit_on_error)
    
    def format_help(self):
        formatter = self._get_formatter()

        # ASCII art, version and script name
        formatter.add_text(f'{text_blink}{textwrap.dedent(ascii_art)}{reset}{fg_pink}{self.prog}{reset}')

        # description
        formatter.add_text(textwrap.indent(self.description, '    ')) # NOTE: add indentation

        # usage
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()
    
    def parse_known_args(self, args=None, namespace=None):
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)
        
        # NOTE: print help if no arguments are given
        if len(args) == 0:
            self.print_help()
            super().exit()

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = argparse.Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not argparse.SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not argparse.SUPPRESS:
                        setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        if self.exit_on_error:
            try:
                namespace, args = self._parse_known_args(args, namespace)
            except argparse.ArgumentError as err:
                self.error(str(err))
        else:
            namespace, args = self._parse_known_args(args, namespace)

        if hasattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR):
            args.extend(getattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR)
        return namespace, args

def setup_parser(description: str, args: list, add_force: bool=False, add_verbose: bool=False) -> argparse.Namespace:
    parser = ArgumentParser(description=description)
    # specific arguments
    for arg in args:
        parser.add_argument(*arg[0], **arg[1])
    # common arguments
    if add_force:
        parser.add_argument('--force', '-f', action='store_true', help='Force overwriting of the output')
    if add_verbose:
        parser.add_argument('--verbose', '-v', type=int, default=2, metavar='VERBOSE_LEVEL', help='Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit')
    return parser.parse_args()

# Progress bar
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
        self._done = False

        if self.total is None:
            bar_length = int(self.ncols // 2)
            self._steps = []
            for i in range(self.ncols - bar_length + 1):
                self._steps.append(f"{fg_black}{'━' * i}{fg_magenta}{'━' * bar_length}{fg_black}{'━' * (self.ncols - bar_length - i)}")
            for i in range(bar_length - 1):
                self._steps.append(f"{fg_magenta}{'━' * (i + 1)}{fg_black}{'━' * (self.ncols - bar_length)}{fg_magenta}{'━' * (bar_length - i - 1)}")
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
                print(f"\r   {step}{reset}", end='', flush=True)
                sleep(self.refresh)
        else:
            while True:
                if self._done:
                    break
                if time() - self._last_time > self.eta_refresh:
                    self._update_eta()
                if self.multithread_progress is not None:
                    self._progress = np.sum(self.multithread_progress)
                print(f"\r   {fg_magenta}{'━' * int(self.ncols * self._progress / self.total)}{fg_black}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {fg_green}{100 * self._progress / self.total:.1f}% {fg_cyan}{self._eta}{reset}", end='', flush=True)
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
                    print(f"\r   {fg_green}{'━' * self.ncols} 100.0%{reset}")
                else:
                    if self.multithread_progress is not None:
                        self._progress = np.sum(self.multithread_progress)
                    print(f"\r   {fg_green}{'━' * int(self.ncols * self._progress / self.total)}{'━' * (self.ncols - int(self.ncols * self._progress / self.total))} {100 * self._progress / self.total:.1f}%{reset}")

    def update(self):
        self._progress += 1

# verbosity level of logging functions
__UI_VERBOSE_LEVEL__ = 4

def set_verbose(verbose: int) -> NoReturn:
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
	if type(verbose) != int or verbose not in [0, 1, 2, 3, 4]:
		raise TypeError('"verbose" must be either 0, 1, 2, 3 or 4')
	__UI_VERBOSE_LEVEL__ = verbose

def get_verbose():
    return __UI_VERBOSE_LEVEL__

def PRINT(*args, **kwargs):
    if __UI_VERBOSE_LEVEL__ >= 3:
        print(*args, **kwargs)

def INFO(message: str) -> NoReturn:
    """Print a INFO message in blue.
    Only shown if __UI_VERBOSE_LEVEL__ >= 3.

    Parameters
    ----------
    message : string
    Message to display.
    """
    if __UI_VERBOSE_LEVEL__ >= 3:
        print(f'{bg_cyan}{fg_black}[ INFO ]{reset} {message}')

def LOG(message: str) -> NoReturn:
    """Print a INFO message in green, reporting the time as well.
    Only shown if __UI_VERBOSE_LEVEL__ >= 3.

    Parameters
    ----------
    message : string
    Message to display.
    """
    if __UI_VERBOSE_LEVEL__ >= 3:
        datetime_str = _datetime.now().strftime('%H:%M:%S')
        print(f'{bg_green}{fg_black}[ {datetime_str} ]{reset} {message}')

def WARNING(message: str, stop: bool=False) -> NoReturn:
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
        print(f'{bg_yellow}{fg_black}[ WARNING ]{reset} {message}')
        if stop:
            sys.exit()

def ERROR(message: str, stop: bool=True) -> NoReturn:
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
        print(f'{bg_red}{fg_black}[ ERROR ]{reset} {message}')
    if stop:
        sys.exit()
