#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from dicelib.tractogram.clustering import cluster
import dicelib.ui as ui

DESCRIPTION = """Cluster a tractogram with QuickBundles"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("--thresholds", "-t", type=float, nargs='+', metavar="THR", help="Threshold(s) [in mm]")
    parser.add_argument("--reference", "-r", action="store", help="Space attributes used as reference for the input tractogram")
    parser.add_argument("--n_pts", type=int, default=12, help="Number of points for the resampling of a streamline")
    parser.add_argument("--replace_centroids", action="store_true", help="Replace centroids with closer streamline in a cluster")
    parser.add_argument("--random", action="store_true", help="Random shuffling of input streamlines")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser


def check_extension(in_arg, out_arg, ref_arg, parser):
    if not in_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        ui.ERROR( 'Invalid input tractogram format' )
    elif not out_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        ui.ERROR( 'Invalid input tractogram format' )
    elif ref_arg is not None and not ref_arg.endswith(('.nii', 'nii.gz')):
        ui.ERROR( 'Invalid reference format' )


def main():
    parser = input_parser()
    options = parser.parse_args()
    
    # check input
    if not os.path.isfile(options.input_tractogram):
        ui.ERROR( f'File "{options.input_tractogram}" not found' )
    if os.path.isfile(options.output_tractogram) and not options.force:
        ui.ERROR("Output tractogram already exists, use -f to overwrite")
    if options.reference is not None:
        if not os.path.isfile(options.reference):
            ui.ERROR( f'File "{options.reference}" not found' )
    check_extension(options.input_tractogram, options.output_tractogram, options.reference, parser)

    # run code
    cluster(
        options.input_tractogram,
        options.reference,
        options.output_tractogram,
        options.thresholds,
        n_pts=options.n_pts,
        replace_centroids=options.replace_centroids,
        random=options.random,
        verbose=options.verbose
    )

if __name__ == "__main__":
    main()
