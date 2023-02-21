#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster, split_clusters
from dicelib.tractogram import split
from dicelib.connectivity import assign
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor as tdp
import concurrent.futures as cf
import os

# parse the input parameters
parser = ColoredArgParser( description=cluster.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("--reference", "-r", help="Reference used for space transofrmation")
parser.add_argument("--atlas", "-a", help="Atlas used to compute streamlines connectivity")
parser.add_argument("--threshold", "-t", type=float, metavar="THR", help="Threshold [in mm]")
parser.add_argument("--n_pts", type=int, default=10, help="Number of points for the resampling of a streamline")
parser.add_argument("--save_assignments", help="Save the cluster assignments to file")
parser.add_argument("--split", action="store_true", help="Split clusters into separate files")
parser.add_argument("--output_folder", "-out", help="Folder where to save the split clusters")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
options = parser.parse_args()

if options.atlas:
    t0 = time.time()
    assign(options.input_tractogram, reference=options.reference, gm_map_file=options.atlas, out_assignment=options.save_assignments, threshold=options.threshold, force=options.force)
    t1 = time.time()
    print("Time taken for connectivity: ", (t1-t0))
else:
    t0 = time.time()
    cluster_idx, _ = cluster(options.input_tractogram,
                        threshold=options.threshold,
                        n_pts=options.n_pts,
                        save_assignments=options.save_assignments,
                        split=options.split,
                        output_folder=options.output_folder,
                        force=options.force,
                        verbose=options.verbose
    )

    t1 = time.time()
    print("Time endin points splitting: ", (t1-t0))
    num_clust = len(np.unique(cluster_idx))
    print(num_clust)

if options.split:
    if options.atlas:
        split(input_tractogram=options.input_tractogram, input_assignments=options.save_assignments, output_folder=options.output_folder, force=options.force)
    else:
        split_clusters(options.input_tractogram, cluster_idx, options.output_folder)
# # if options.save_assignments:
# #     np.savetxt(options.save_assignments, cluster_idx)

MAX_THREAD = 6
executor = tdp(max_workers=MAX_THREAD)
bundles = []
res_parallel = []
for dirpath,_,filenames in os.walk(options.output_folder):
    for f in filenames:
        if f.endswith('.tck'):
            bundles.append( os.path.abspath(os.path.join(dirpath, f)))

options.threshold = 2
options.n_pts = 10

t0 = time.time()
future = [executor.submit(cluster, bundles[i], 
                        threshold=options.threshold,
                        n_pts=options.n_pts,
                        save_assignments=options.save_assignments,
                        # split=options.split,
                        output_folder=options.output_folder,
                        force=options.force,
                        verbose=options.verbose) for i in range(len(bundles))]

# for f in future:
for i, f in enumerate(cf.as_completed(future)):
    print(f"Done: {i}/{len(bundles)}", end="\r")
    res_parallel.append(f.result())
t1 = time.time()
print()
print("Time taken for parallel: ", (t1-t0)/60)
# call actual function


