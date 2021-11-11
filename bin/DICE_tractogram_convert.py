#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import set_sft_logger_level
set_sft_logger_level("CRITICAL")

DESCRIPTION = """Tractogram conversion from and to '.tck', '.trk', '.fib',
             '.vtk' and 'dpy'. All the extensions except '.trk, need a NIFTI
             file as reference """


def input_parser():
    parser = argparse.ArgumentParser(usage="%(prog)s Input_tractogram "
                                     "Output_tractogram -r Reference",
                                     description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("-r", action="store", dest="reference",
                        help="Space attributes used as \n"
                        "reference for the input tractogram")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwriting of the output")
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


def load_inp(in_arg, ref=None):
    if ref is None:
        in_reference = "same"
        try:
            inp_tract = load_tractogram(in_arg,
                                        reference=in_reference,
                                        bbox_valid_check=True,
                                        trk_header_check=True)
        except ValueError:
            raise
    else:
        in_reference = ref
        try:
            inp_tract = load_tractogram(in_arg,
                                        reference=in_reference,
                                        # "to_space=Space.RASMM,"
                                        # "to_origin=Origin.NIFTI,"
                                        bbox_valid_check=True)
        except ValueError:
            raise
    return inp_tract


def main():
    parser = input_parser()
    p_args = parser.parse_args()
    check_path(p_args, parser)
    check_extension(p_args.input_tractogram, p_args.output_tractogram,
                    p_args.reference, parser)

    if p_args.reference is None:
        if p_args.input_tractogram.endswith(".trk"):
            try:
                sft_in = load_inp(p_args.input_tractogram)
            except Exception:
                raise
        else:
            parser.error("reference is required if the input format is '.tck'")
    else:
        try:
            sft_in = load_inp(p_args.input_tractogram, ref=p_args.reference)
        except Exception:
            raise
    try:
        save_tractogram(sft_in, p_args.output_tractogram)
    except (OSError, TypeError) as e:
        parser.error("Output not valid: {}".format(e))


if __name__ == "__main__":
    main()
