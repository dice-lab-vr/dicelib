#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import spline_smoothing

# parse the input parameters
parser = ColoredArgParser( description=spline_smoothing.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument("--ratio", "-r", type=float, default=0.25, help="Ratio of points to be kept/used as control points")
parser.add_argument("--step", "-s", type=float, default=1.0, help="Sampling step for the output streamlines [in mm]")
parser.add_argument("--verbose", "-v", type=int, default=2, help="What information to print (must be in [0...4] as defined in ui)")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
spline_smoothing(
    options.input_tractogram,
    options.output_tractogram,
    options.ratio,
    options.step,
    options.verbose,
    options.force
)
