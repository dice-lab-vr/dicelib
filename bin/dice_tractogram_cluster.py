#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.clustering import run_clustering

# parse the input parameters
parser = ColoredArgParser( description=run_clustering.__doc__.split('\n')[0] )
parser.add_argument("file_name_in", help="Input tractogram")
parser.add_argument("--atlas", "-a", help="Atlas used to compute streamlines connectivity")
parser.add_argument("--conn_thr", "-t", default=2, type=float, metavar="THR", help="Threshold [in mm]")
parser.add_argument("--clust_thr", type=float, help="Threshold [in mm]")
parser.add_argument("--n_pts", type=int, default=10, help="Number of points for the resampling of a streamline")
parser.add_argument("--save_assignments", "-s", help="Save the cluster assignments to file")
parser.add_argument("--output_folder", "-out", help="Folder where to save the split clusters")
parser.add_argument("--file_name_out", "-o", default=None, help="Output clustered tractogram")
parser.add_argument("--n_threads", type=int, help="Number of threads to use to perform clustering")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", default=2, type=int, help="Verbose level [ 0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress ]")
options = parser.parse_args()


def main():
    run_clustering(file_name_in=options.file_name_in, output_folder=options.output_folder, file_name_out=options.file_name_out, atlas=options.atlas, conn_thr=options.conn_thr, clust_thr=options.clust_thr, n_pts=options.n_pts,
                   save_assignments=options.save_assignments, n_threads=options.n_threads, force=options.force, verbose=options.verbose)
    
if __name__ == "__main__":
    main()