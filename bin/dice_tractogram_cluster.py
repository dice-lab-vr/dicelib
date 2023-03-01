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
parser.add_argument("--n_threads", type=int, help="Number of threads to use to perform clustering")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
options = parser.parse_args()


def compute_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



tt0 = time.time()

if options.n_threads:
    MAX_THREAD = options.n_threads
else:
    MAX_THREAD = 1


num_streamlines = int(nib.streamlines.load(options.input_tractogram, lazy_load=True).header["count"])
print(num_streamlines)
chunk_size = int(num_streamlines/MAX_THREAD)
# print(chunk_size)
chunk_groups = [e for e in compute_chunks( np.arange(num_streamlines),chunk_size)]
print([len(c) for c in chunk_groups])

# chunk_size = [[end_chunks[i] - end_chunks[i-1]] for i in range(1,len(end_chunks))]


if options.atlas:
    chunks_asgn = []
    t0 = time.time()
    # future = [executor.submit(assign, input_tractogram=options.input_tractogram, start_chunk =chunk_groups[i][0], end_chunk=chunk_groups[i][len(chunk_groups[i])-1], chunk_size=len(chunk_groups[i]),
    #                         reference=options.reference, gm_map_file=options.atlas, out_assignment=options.save_assignments,
    #                         threshold=options.threshold, force=options.force) for i in range(len(chunk_groups))]
    # chunks_asgn = assign(input_tractogram=options.input_tractogram, start_chunk =0, end_chunk=num_streamlines, chunk_size=num_streamlines,
    #                         reference=options.reference, gm_map_file=options.atlas, out_assignment=options.save_assignments,
    #                         threshold=options.threshold, force=options.force)
    with tdp(max_workers=MAX_THREAD) as executor:
        future = [executor.submit(assign, input_tractogram=options.input_tractogram, start_chunk =chunk_groups[i][0], end_chunk=chunk_groups[i][len(chunk_groups[i])-1]+1, chunk_size=len(chunk_groups[i]),
                            reference=options.reference, gm_map_file=options.atlas, out_assignment=options.save_assignments,
                            threshold=options.threshold, force=options.force) for i in range(len(chunk_groups))]
    # for i, f in enumerate(future):
    chunks_asgn = [f.result() for f in future]
    chunks_asgn = [c for f in chunks_asgn for c in f]
    # print(chunks_asgn[:10])
    # for i, f in enumerate(cf.as_completed(future)):
    #     print(f"Done chunk: {i}/{len(chunk_groups)}")
    #     chunks_asgn.extend(f.result())
    print("Done")
    t1 = time.time()
    print("Time taken for connectivity: ", (t1-t0))
    out_assignment_ext = os.path.splitext(options.save_assignments)[1]
    # out_assignment = f"{out_assignment[:-4]}{out_assignment_ext}"
    if out_assignment_ext not in ['.txt', '.npy']:
        print( 'Invalid extension for the output scalar file' )
    if os.path.isfile(options.save_assignments) and not options.force:
        print( 'Output scalar file already exists, use -f to overwrite' )
    if out_assignment_ext=='.txt':
        with open(options.save_assignments, "w") as text_file:
            for reg in chunks_asgn:
                print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
    else:
        np.save( options.save_assignments, chunks_asgn, allow_pickle=False )

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
bundles = []
res_parallel = []
res_single = []
options.threshold = 2
options.n_pts = 10

centroids_list = []

for  dirpath, _, filenames in os.walk(options.output_folder):
    for i_b, f in enumerate(filenames):
        if f.endswith('.tck') and not f.startswith('unassigned'):
            bundles.append(os.path.abspath(os.path.join(dirpath, f)))

#TODO sorted_files = sorted(all_files, key = os.path.getsize)
# 1. Order the available items descending.
# 2. Create N empty groups
# 3. Start adding the items one at a time into the group that has the smallest size sum in it.

executor = tdp(max_workers=MAX_THREAD)
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
