#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
from dicelib.lazytractogram import LazyTractogram
from Tsf import Tsf
from dicelib.ui import ColoredArgParser

# create script to color streamlines using mrtrix
def create_color_scalar_file(streamline, num_streamlines):
    """
    Create a scalar file for each streamline in order to color them.
    Parameters
    ----------
    streamlines: list
        List of streamlines.
    Returns
    -------
    scalar_file: str
        Path to scalar file.
    """
    scalar_list = list()
    n_pts_list = list()
    for i in range(num_streamlines):
        # pt_list = list()
        streamline.read_streamline()
        n_pts_list.append(streamline.n_pts)
        for j in range(streamline.n_pts):
            scalar_list.extend([float(j)])
        # scalar_list.append(pt_list)
    return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)

def color_by_scalar_file(streamlines, values, num_streamlines):
    """
    Color streamlines based on sections.
    Parameters
    ----------
    streamlines: array
        Array of streamlines.
    values: list
        List of scalars.
    Returns
    -------
    array
        Array mapping scalar values to each vertex of each streamline.
    array
        Array containing the number of points of each input streamline.
    """
    scalar_list = []
    n_pts_list = []
    for i in range(num_streamlines):
        streamline.read_streamline()
        n_pts_list.append(streamline.n_pts)
        streamline_points = np.arange(streamline.n_pts)
        resample = np.linspace(0, streamline.n_pts, len(values), endpoint=True, dtype=np.int32)
        streamline_points = np.interp(streamline_points, resample, values)
        scalar_list.extend(streamline_points)
    return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)



# parse the input parameters
parser = ColoredArgParser( description=filter.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("output_tsf", help="Output tsf filenam")
parser.add_argument("--orientation", action="store_true", default=True, help="Color based on orientation")
parser.add_argument("--file", action="store", help="Color based on given file")
parser.add_argument("--force", "-f",   action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

streamline = LazyTractogram( options.input_tractogram, mode='r')
num_streamlines = streamline.header['count']

if options.orientation:
    scalar_arr, n_pts_list = create_color_scalar_file(streamline, int(num_streamlines))
elif options.file:
    values = np.loadtxt(options.file)
    scalar_arr, n_pts_list = color_by_scalar_file(streamline, values, int(num_streamlines))
else:
    raise ValueError("Please specify a color option")

#check if output file exists
if os.path.isfile(options.output_tsf) and not options.force:
    raise IOError("Output file already exists. Use -f to overwrite.")

tsf = Tsf(options.output_tsf, 'w', header=streamline.header)
tsf.write_scalar(scalar_arr, n_pts_list)