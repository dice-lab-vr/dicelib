#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import info

# parse the input parameters
parser = ColoredArgParser( description=info.__doc__.split('\n')[0] )
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("--lenghts", "-l", action="store_true", help="Show stats on streamline lenghts")
parser.add_argument("--max_field_length", "-m", type=int, help="Maximum length allowed for printing a field value")
parser.add_argument("--verbose", "-v", default=2, type=int, help="Verbose level [ 0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress ]")
options = parser.parse_args()

# call actual function
info(
    options.tractogram,
    options.lenghts,
    options.max_field_length,
    options.verbose
)