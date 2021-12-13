#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys, os
from dicelib.tractogram import split

# parse the input parameters
parser = argparse.ArgumentParser(
    description="Split the streamlines in a tractogram according to a (precomputed) assignment file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("tractogram", help="Input tractogram")
parser.add_argument("assignments", help="Text file with the streamline assignments")
parser.add_argument("output_folder", nargs='?', default='bundles', help="Output folder for the splitted tractograms")
parser.add_argument("--max_open", "-m", type=int, help="Maximum number of files open at the same time")
parser.add_argument("--verbose", "-v", action="store_true", help="Print information messages")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
options = parser.parse_args()

# call actual function
split(
    options.tractogram,
    options.assignments,
    options.output_folder,
    options.max_open,
    options.verbose,
    options.force
)