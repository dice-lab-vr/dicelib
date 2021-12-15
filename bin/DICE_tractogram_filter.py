#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import filter

# parse the input parameters
parser = ColoredArgParser( description="Filter out the streamlines in a tractogram according to some criteria." )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument("--minlength", type=float, help="Keep streamlines with length [in mm] >= this value")
parser.add_argument("--maxlength", type=float, help="Keep streamlines with length [in mm] <= this value")
parser.add_argument("--minweight", type=float, help="Keep streamlines with weight >= this value")
parser.add_argument("--maxweight", type=float, help="Keep streamlines with weight <= this value")
parser.add_argument("--weights_in",  help="Text file with the input streamline weights")
parser.add_argument("--weights_out", help="Text file for the output streamline weights")
parser.add_argument("--verbose", "-v", action="store_true", help="Print information messages")
parser.add_argument("--force", "-f",   action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
filter(
    options.input_tractogram,
    options.output_tractogram,
    options.minlength,
    options.maxlength,
    options.minweight,
    options.maxweight,
    options.weights_in,
    options.weights_out,
    options.verbose,
    options.force
)