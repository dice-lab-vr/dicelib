#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys
from dicelib.tractogram import info

# parse the input parameters
parser = argparse.ArgumentParser(
    description="Print some information about a tractogram",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
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