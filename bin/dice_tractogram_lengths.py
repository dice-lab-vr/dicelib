#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import compute_lenghts

# parse the input parameters
parser = ColoredArgParser( description=compute_lenghts.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_scalar_file", help="Output scalar file (.npy or .txt) that will contain the streamline lengths")
parser.add_argument("--verbose", "-v", type=int, default=2, help="What information to print (must be in [0...4] as defined in ui)")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
compute_lenghts(
    options.input_tractogram,
    options.output_scalar_file,
    options.verbose,
    options.force
)