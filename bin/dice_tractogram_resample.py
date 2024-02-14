#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import dicelib.ui as ui
from dicelib.tractogram import tractogram_resample
from dicelib.ui import ColoredArgParser

# parse the input parameters
parser = ColoredArgParser(description=tractogram_resample.__doc__.split('\n')[0])
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("output", help="Output tractogram")
parser.add_argument("--nb_points", "-n", type=int, default=20,
                    help="Number of points per streamline")
parser.add_argument("--force", "-f", action="store_true",
                    help="Overwrite output file if it already exists")
parser.add_argument(
    "--verbose",
    "-v",
    default=2,
    type=int,
    help=("Verbose level [ 0 = no output, 1 = only errors/warnings, "
          "2 = errors/warnings and progress, 3 = all messages, no progress, "
          "4 = all messages and progress ]")
)

options = parser.parse_args()

# check if path to input and output files are valid
if not os.path.isfile(options.tractogram):
    ui.ERROR(f"Input tractogram file not found: {options.tractogram}")
if os.path.isfile(options.output):
    if options.force:
        ui.WARNING(f"Overwriting output file: {options.output}")
    else:
        ui.ERROR(f"Output file already exists: {options.output}")
if options.nb_points < 2:
    ui.ERROR(f"Number of points per streamline must be >= 2: {options.nb_points}")

# call actual function
tractogram_resample(
    options.tractogram,
    options.output,
    options.nb_points,
    options.verbose,
    options.force,
)        
