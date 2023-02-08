#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster, run_cluster_parallel
from geom_clustering import split_clusters
import numpy as np

# parse the input parameters
parser = ColoredArgParser( description=cluster.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tractogram", help="Output tractogram")
parser.add_argument("--reference", default=None, help="NIFTI file used as reference for the input tractogram")
parser.add_argument("--threshold", "-t", type=float, metavar="THR", help="Threshold [in mm]")
parser.add_argument("--n_pts", type=int, default=10, help="Number of points for the resampling of a streamline")
parser.add_argument("--replace_centroids", action="store_true", help="Replace centroids with closer streamline in a cluster")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
options = parser.parse_args()

# call actual function
# cluster_idx = cluster(options.input_tractogram,
#                     options.output_tractogram,
#                     options.reference,
#                     options.threshold,
#                     n_pts=options.n_pts,
#                     replace_centroids=options.replace_centroids,
#                     force=options.force,
#                     verbose=options.verbose
# )

cluster_idx = run_cluster_parallel(options.input_tractogram,
                    options.output_tractogram,
                    options.reference,
                    options.threshold,
                    n_pts=options.n_pts,
                    replace_centroids=options.replace_centroids,
                    force=options.force,
                    verbose=options.verbose
)

output_folder = "/home/matteo/Dataset/HCP_100307/Tractography/bundles"

split_clusters(options.input_tractogram, np.array(cluster_idx), output_folder)
# import numpy as np
# np.savetxt("/home/matteo/Dataset/HCP_100307/Tractography/cluster_by_colot.txt", np.array(cluster_idx))