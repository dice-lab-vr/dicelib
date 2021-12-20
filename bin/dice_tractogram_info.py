#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import info

# parse the input parameters
parser = ColoredArgParser( description=info.__doc__.split('\n')[0] )
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("--lenghts", "-l", action="store_true", help="Show stats on streamline lenghts")
options = parser.parse_args()

# call actual function
info(
    options.tractogram,
    options.lenghts
)