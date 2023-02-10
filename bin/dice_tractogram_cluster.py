#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster#, run_cluster_parallel
from geom_clustering import split_clusters
import numpy as np
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.io.streamline import load_tractogram, save_tractogram, Space
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import time
import nibabel as nib

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
cluster_idx = cluster(options.input_tractogram,
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

# print("Clustering with Quickbundle")
# output_folder_quickbund = "/home/matteo/Dataset/HCP_100307/Tractography/bundles_quickbund"
# ref = "/home/matteo/Dataset/HCP_100307/wm.nii.gz" 
# tractogram = options.input_tractogram

# t1 = time.time()
# sft_tractogram = load_tractogram(tractogram, ref, bbox_valid_check=False)
# feature = ResampleFeature(nb_points=options.n_pts)
# metric = AveragePointwiseEuclideanMetric(feature)
# streamlines = sft_tractogram.streamlines

# print(f"time required for loading: {np.round((time.time()-t1)/60, 3)} minutes")

# qb = QuickBundles(threshold=options.threshold, metric=metric)
# clusters = qb.cluster(streamlines)
# print(f"time required Quickbundles: {np.round((time.time()-t1)/60, 3)} minutes")
# print(f"number of clusters Quickbundles {len(clusters)}")
# clust_ass = np.zeros(len(sft_tractogram.streamlines))
# for i in range(len(clusters)):
#     for v in clusters[i].indices:
#         clust_ass[v] = i

# split_clusters(tractogram, clust_ass, output_folder_quickbund)