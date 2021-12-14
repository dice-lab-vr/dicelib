#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from dicelib.tractogram import info
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser( description="Print some information about a tractogram." )
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("--lenghts", "-l", action="store_true", help="Show stats on streamline lenghts")
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
options = parser.parse_args()

# call actual function
info(
    options.tractogram,
    options.lenghts
)