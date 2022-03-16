#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dicelib.ui import ColoredArgParser
from dicelib.connectivity import assign
import numpy as np
import time
import os

parser = ColoredArgParser( description=assign.__doc__.split('\n')[0] )
parser.add_argument("input_tractogram", help="Input tractogram")
parser.add_argument("gm_map_file", help="NIFTI file containing the ROIs")
parser.add_argument("out_assignment", help="Output assignment file")
parser.add_argument("--threshold", "-t", type=float, metavar="THR", help="Threshold [in mm]")
parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
options = parser.parse_args()

start = time.time()
assign(
    options.input_tractogram,
    options.gm_map_file,
    options.out_assignment,
    options.threshold,
    verbose=options.verbose,
    force=options.force
)
print(f"test DICE radial search time: {time.time()- start}")

mtrix_ass = f"/media/full/DATA/PhD_Data/Phantomas/ISBI2013_FINAL_version/mrtrix_assignment_radial_search_{options.threshold}.txt"
start = time.time()
cmd = f"tck2connectome {options.input_tractogram} {options.gm_map_file} -assignment_radial_search {options.threshold} connectome.csv -out_assignments {mtrix_ass} -force -quiet"
os.system(cmd)
print(f"mrtrix time radial search: {time.time()- start}")

# TESTING differences with mrtrix
dice_conn = np.loadtxt(f"/media/full/DATA/PhD_Data/Phantomas/ISBI2013_FINAL_version/test_assignment_radial_search_{options.threshold}.txt")
diff_assignments_out = f"/media/full/DATA/PhD_Data/Phantomas/ISBI2013_FINAL_version/test_different_assignment_radial_search_{options.threshold}.txt"
mrtrix_conn = np.loadtxt(mtrix_ass)
count = 0
with open(diff_assignments_out, 'w') as diff_file:
    for i,j in zip(dice_conn, mrtrix_conn):
        if sorted(i) != sorted(j):
            count += 1
            print(f"{int(i[0])} {int(i[1])} <-> {int(j[0])} {int(j[1])}", file=diff_file)
print(f"Different streamlines connectivity: {count}")