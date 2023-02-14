#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster
import numpy as np
import time

# parse the input parameters
parser = ColoredArgParser( description=cluster.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("--threshold", "-t", type=float, metavar="THR", help="Threshold [in mm]")
parser.add_argument("--n_pts", type=int, default=10, help="Number of points for the resampling of a streamline")
parser.add_argument("--save_assignments", help="Save the cluster assignments to file")
parser.add_argument("--split", action="store_true", help="Split clusters into separate files")
parser.add_argument("--output_folder", help="Folder where to save the split clusters")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
options = parser.parse_args()

# call actual function
cluster_idx = cluster(options.input_tractogram,
                    threshold=options.threshold,
                    n_pts=options.n_pts,
                    save_assignments=options.save_assignments,
                    split=options.split,
                    output_folder=options.output_folder,
                    force=options.force,
                    verbose=options.verbose
)
