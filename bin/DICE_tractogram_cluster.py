#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from dicelib.tractogram.clustering import cluster

DESCRIPTION = """Cluster a tractogram with QuickBundles"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("threshold", type=float, help="Threshold [in mm]")
    parser.add_argument("--n_pts", type=int, default=12, help="Number of points for the resampling of a streamline")
    parser.add_argument("--reference", "-r", action="store", help="Space attributes used as reference for the input tractogram")
    parser.add_argument("--replace_centroids", action="store_true", help="Replace centroids with closer streamline in a cluster")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser


def check_extension(in_arg, out_arg, ref_arg, parser):
    if not in_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        parser.error("Invalid input tractogram format")
    elif not out_arg.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        parser.error("Invalid input tractogram format")
    elif ref_arg is not None and not ref_arg.endswith(('.nii', 'nii.gz')):
        parser.error("Invalid reference format")


def check_path(args, parser):
    in_file = args.input_tractogram
    out_file = args.output_tractogram
    in_ref = args.reference

    if not os.path.isfile(in_file):
        parser.error("No such file {}".format(in_file))
    if os.path.isfile(out_file) and not args.force:
        parser.error("Output tractogram already exists, use -f to overwrite")
    if in_ref is not None:
        if not os.path.isfile(args.reference):
            parser.error("No such file {}".format(args.reference))


def main():
    parser = input_parser()
    p_args = parser.parse_args()
    check_path(p_args, parser)
    check_extension(p_args.input_tractogram, p_args.output_tractogram, p_args.reference, parser)

    cluster(
        p_args.input_tractogram,
        p_args.reference,
        p_args.output_tractogram,
        [p_args.threshold],
        n_pts=p_args.n_pts,
        replace_centroids=p_args.replace_centroids,
        random=True,
        verbose=False
    )

if __name__ == "__main__":
    main()
