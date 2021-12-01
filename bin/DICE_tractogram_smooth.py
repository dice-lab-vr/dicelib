#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
from dicelib.tractogram.processing import spline_smoothing
import dicelib.ui as ui

DESCRIPTION = """Smooth the streamlines in a tractogram using Catmull-Rom splines"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("--ratio", "-r", type=float, default=0.25, help="Ratio of points to be kept/used as control points")
    parser.add_argument("--step", "-s", type=float, default=1.0, help="Sampling step for the output streamlines [in mm]")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser


def main():
    parser = input_parser()
    options = parser.parse_args()

    # check input
    if not os.path.isfile(options.input_tractogram):
        ui.ERROR( f'File "{options.input_tractogram}" not found' )
    if os.path.isfile(options.output_tractogram) and not options.force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )

    # run code
    spline_smoothing(
        options.input_tractogram,
        filename_tractogram_out=options.output_tractogram,
        control_point_ratio=options.ratio,
        segment_len=options.step,
        verbose=options.verbose
    )


if __name__ == "__main__":
    main()
