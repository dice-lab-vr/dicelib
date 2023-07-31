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
parser.add_argument("--save_assignments", help="Save the cluster assignments to file")
parser.add_argument("--output_folder", "-out", help="Folder where to save the split clusters")
parser.add_argument("--n_threads", type=int, help="Number of threads to use to perform clustering")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
parser.add_argument("--remove_outliers", "-ro", default=False, action="store_true", help="Remove outliers - beta feature")
options = parser.parse_args()


def main():
    run_clustering(options.file_name_in, options.output_folder, options.atlas, options.reference,
                   options.conn_thr, options.clust_thr, options.n_pts, options.save_assignments,
                   options.split, options.n_threads, options.remove_outliers, options.force, options.verbose)
    
if __name__ == "__main__":
    main()