#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pickletools import int4
from numpy import float32
from dicelib.ui import ColoredArgParser
from dicelib.create_structure_2_levels import create_structure_2_levels

# parse the input parameters
parser = ColoredArgParser( description=create_structure_2_levels.__doc__.split('\n')[0] )
parser.add_argument("filename_trk", help="Input tractogram")
parser.add_argument("filename_cm", help="Input connectome")
parser.add_argument("filename_fs", help="Text file with the streamline assignments")
parser.add_argument("output_folder", help="Output folder for the structures")
parser.add_argument("--QB_threshold", type=float32, default=10, help=" Threstold to use in QuickBundles")
parser.add_argument("--sufix",default="", help="Sufix to name the output files")
parser.add_argument("--verbose", "-v", type=int, default=4, help="What information to print (must be in [0...4] as defined in ui)")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()


# call actual function
create_structure_2_levels(
    options.filename_trk,
    options.filename_cm,
    options.filename_fs,
    options.output_folder,
    QB_threshold=options.QB_threshold,
    sufix=options.sufix,
    metric=None,
    verbose=options.verbose,
    force=options.force
)