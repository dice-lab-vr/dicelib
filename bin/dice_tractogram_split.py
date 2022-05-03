#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import split

# parse the input parameters
parser = ColoredArgParser( description=split.__doc__.split('\n')[0] )
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("assignments", help="Text file with the streamline assignments")
parser.add_argument("output_folder", nargs='?', default='bundles', help="Output folder for the splitted tractograms")
parser.add_argument("--weights_in",  help="Text file with the input streamline weights")
parser.add_argument("--max_open", "-m", type=int, help="Maximum number of files opened at the same time")
parser.add_argument("--verbose", "-v", type=int, default=2, help="What information to print (must be in [0...4] as defined in ui)")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
split(
    options.tractogram,
    options.assignments,
    options.output_folder,
    options.weights_in,
    options.max_open,
    options.verbose,
    options.force
)