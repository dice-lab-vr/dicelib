#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.dwi import extract

# parse the input parameters
parser = ColoredArgParser( description=extract.__doc__.split('\n')[0] )
parser.add_argument("input_dwi", help="Input DWI data")
parser.add_argument("input_scheme", help="Input scheme")
parser.add_argument("output_dwi", help="Output DWI data")
parser.add_argument("output_scheme", help="Output scheme")
parser.add_argument("--b", "-b", type=float, nargs='+', required=True, help="List of b-values to extract")
parser.add_argument("--round", "-r", type=float, default=0.0, help="Round b-values to nearest integer multiple of this value")
parser.add_argument("--verbose", "-v", action="store_true", help="Print information messages")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
extract(
    options.input_dwi,
    options.input_scheme,
    options.output_dwi,
    options.output_scheme,
    options.b,
    options.round,
    options.verbose,
    options.force
)