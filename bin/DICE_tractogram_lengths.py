#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys
from dicelib.tractogram import compute_lenghts

# parse the input parameters
parser = argparse.ArgumentParser(
    description="Return the length in mm of the streamlines in a tractogram",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_scalar_file", help="Output scalar file that will contain the streamline lengths")
parser.add_argument("--verbose", "-v", action="store_true", help="Print information messages")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
options = parser.parse_args()

# call actual function
compute_lenghts(
    options.input_tractogram,
    options.output_scalar_file,
    options.verbose,
    options.force
)