#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from dicelib.clustering import tractogram_cluster

DESCRIPTION = """Cluster a tractogram with QuickBundles"""

def input_parser():
    parser = argparse.ArgumentParser(usage="%(prog)s <input_tractogram> <output_tractogram> <threshold> -r <reference>", description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("threshold", type=float, help="Threshold [in mm]")
    parser.add_argument("-r", action="store", dest="reference", help="Space attributes used as reference for the input tractogram")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwriting of the output")
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

    tractogram_cluster(
        p_args.input_tractogram,
        p_args.reference,
        p_args.output_tractogram,
        [p_args.threshold],
        n_pts=20,
        random=True,
        verbose=False
    )

if __name__ == "__main__":
    main()
