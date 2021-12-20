#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster

# parse the input parameters
parser = ColoredArgParser( description=cluster.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("reference", help="NIFTI file used as reference for the input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument("--thresholds", "-t", type=float, nargs='+', metavar="THR", help="Threshold(s) [in mm]")
parser.add_argument("--n_pts", type=int, default=12, help="Number of points for the resampling of a streamline")
parser.add_argument("--replace_centroids", action="store_true", help="Replace centroids with closer streamline in a cluster")
parser.add_argument("--random", action="store_true", help="Random shuffling of input streamlines")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

# call actual function
cluster(
    options.input_tractogram,
    options.reference,
    options.output_tractogram,
    options.thresholds,
    n_pts=options.n_pts,
    replace_centroids=options.replace_centroids,
    random=options.random,
    verbose=options.verbose,
    force=options.force
)
