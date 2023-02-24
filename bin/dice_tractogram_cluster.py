#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import cluster, split_clusters, closest_streamline
from dicelib.tractogram import split
from dicelib.connectivity import assign
import numpy as np
import nibabel as nib
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
tt0 = time.time()
if options.atlas:
    t0 = time.time()
    assign(options.input_tractogram, reference=options.reference, gm_map_file=options.atlas, out_assignment=options.save_assignments, threshold=options.threshold, force=options.force)
    t1 = time.time()
    print("Time taken for connectivity: ", (t1-t0))
else:
    t0 = time.time()
    cluster_idx, _, _ = cluster(options.input_tractogram,
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
t0 = time.time()
if options.split:
    if options.atlas:
        split(input_tractogram=options.input_tractogram, input_assignments=options.save_assignments, output_folder=options.output_folder, force=options.force)
    else:
        split_clusters(options.input_tractogram, cluster_idx, options.output_folder)
# # if options.save_assignments:
# #     np.savetxt(options.save_assignments, cluster_idx)

t1 = time.time()
print("Time bundles splitting: ", (t1-t0))


def cluster_bundle(bundle, threshold=options.threshold, n_pts=options.n_pts, save_assignments=options.save_assignments,
                    output_folder=options.output_folder, force=options.force, verbose=options.verbose):
    clust_idx, set_centroids  = cluster(bundle, 
                            threshold=options.threshold,
                            n_pts=options.n_pts,
                            save_assignments=options.save_assignments,
                            # split=options.split,
                            output_folder=options.output_folder,
                            force=options.force,
                            verbose=options.verbose)
    centr_len = np.zeros(set_centroids.shape[0], dtype=np.intc)
    new_c = closest_streamline(bundle, set_centroids, clust_idx, options.n_pts, set_centroids.shape[0], centr_len)
    return new_c, centr_len



t0 = time.time()
MAX_THREAD = 16
executor = tdp(max_workers=MAX_THREAD)
bundles = []
res_parallel = []
res_single = []
options.threshold = 2
options.n_pts = 10

centroids_list = []

for  dirpath, _, filenames in os.walk(options.output_folder):
    for i_b, f in enumerate(filenames):
        bundles.append(os.path.abspath(os.path.join(dirpath, f)))

t0 = time.time()
future = [executor.submit(cluster_bundle, bundles[i], 
                        threshold=options.threshold,
                        n_pts=options.n_pts,
                        save_assignments=options.save_assignments,
                        # split=options.split,
                        output_folder=options.output_folder,
                        force=options.force,
                        verbose=options.verbose) for i in range(len(bundles))]

for i, f in enumerate(cf.as_completed(future)):
        print(f"Done: {i}/{len(bundles)}", end="\r")
        new_c, centr_len = f.result()
        for jj, n_c in enumerate(new_c):
            centroids_list.append(n_c[:centr_len[jj]])
#
t1 = time.time()
print(f"Time taken to cluster and find closest streamlines: {t1-t0}" )
tt1= time.time() -tt0 
print(f"Total time required for clustering: {tt1}" )
print("Saving centroids...")
ref_data = nib.load(options.reference)
ref_header = ref_data.header
affine = ref_data.affine
centroids_out = nib.streamlines.Tractogram(centroids_list, affine_to_rasmm=affine)

nib.streamlines.save(centroids_out, os.path.join("/home/matteo/Dataset/HCP_test_retest/baseline/172332",'centroids.tck'))
