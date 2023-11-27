#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import dicelib.ui as ui
from dicelib.connectome_blur import build_connectome
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=build_connectome.__doc__.split('\n')[0])
parser.add_argument("input_assignments", help="Input streamline assignments file")
parser.add_argument("output_connectome", help="Output connectome")
parser.add_argument("input_weights", help="Input streamline weights file")
parser.add_argument(
    "--metric",
    "-m",
    default='sum',
    help="Operation to compute the value of the edges, options: sum, mean, min, max.")
parser.add_argument(
    "--symmetric", 
    "-s", 
    action="store_true",
    help="Make output connectome symmetric")
parser.add_argument(
    "--verbose",
    "-v",
    default=2,
    type=int,
    help=("Verbose level [ 0 = no output, 1 = only errors/warnings, "
          "2 = errors/warnings and progress, 3 = all messages, no progress, "
          "4 = all messages and progress ]")
)
parser.add_argument("--force", "-f", action="store_true",
                    help="Force overwriting of the output")
options = parser.parse_args()


# check if path to input and output files are valid
if not os.path.isfile(options.input_assignments):
    ui.ERROR(f"Input assignments file not found: {options.input_assignments}")
if os.path.isfile(options.output_connectome) and not options.force:
    ui.ERROR(
        f"Output conncetome file already exists: {options.output_connectome}")
# check if the output connectome file has the correct extension
output_connectome_ext = os.path.splitext(options.output_connectome)[1]
if output_connectome_ext not in ['.csv', '.npy']:
    ui.ERROR("Invalid extension for the output connectome file")

# check if the output connectome file has absolute path and if not, add the current working directory
if options.output_connectome and not os.path.isabs(options.output_connectome):
    options.output_connectome = os.path.join(os.getcwd(), options.output_connectome)


# call actual function
build_connectome( 
    options.input_assignments,
    options.output_connectome,
    options.input_weights, 
    options.metric, 
    options.symmetric,
    options.verbose,
    options.force
)
