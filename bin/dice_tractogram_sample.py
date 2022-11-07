#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.tractogram import sample 

# parse the input parameters
parser = ColoredArgParser( description=sample.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("input_image", help="Input image")
parser.add_argument("output_file", help="File for the output")
parser.add_argument("--space", nargs='?', default= None, choices=['voxmm','rasmm','vox'] , help="Current reference space of streamlines(rasmm,voxmm,vox),default rasmm")
parser.add_argument("--option", nargs='?', default="No_opt", choices=['No_opt','mean','median','min','max'], help="Operation to apply on streamlines, default no operation applied")
parser.add_argument("--force", "-f",   action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
sample(
    options.input_tractogram,
    options.input_image,
    options.output_file,
    options.space,
    options.option,
    options.force
)