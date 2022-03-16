#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
import nibabel as nib
import sys, os, glob, random
from .lazytck import LazyTCK
from . import ui
from tqdm import trange
from libc.math cimport sqrt

cdef compute_grid(int thr=0):
    x = np.linspace(-thr, thr, 2*thr +1) + .5 
    y = np.linspace(-thr, thr, 2*thr +1) + .5
    z = np.linspace(-thr, thr, 2*thr +1) + .5
    # # create the mesh based on these arrays
    mx, my, mz = np.meshgrid(x, y, z)
    corners = np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1)

    center = np.array([.5, .5, .5 ])
    dist_grid = ((corners - center)**2).sum(axis=1)  # compute distances
    sorted_grid = dist_grid.argsort()
    corners = corners[sorted_grid]
    # print(corners[:10])
    return corners

cpdef streamline_assignment( float [:,:] streamline, int n, double [:,:] grid, gm_map ):
    cdef int roi1 = 0
    cdef int roi2 = 0
    cdef int found1 = 0
    cdef int found2 = 0

    # cdef int n = streamline.shape[0]
    cdef int grid_size = grid.shape[0]

    cdef float* ptr     = &streamline[0,0]
    cdef float* ptr_end = ptr+n*3-3
    # cdef float* ptr_end = ptr+n
    cdef size_t i
    # print(f"start pointer: {(ptr[0], ptr[1], ptr[2])}")
    # print(f"len streamline: {n}")
    # print(f"end pointer: {(ptr_end[0], ptr_end[1], ptr_end[2])}")

    for i in xrange(grid_size):
        # s_pt = np.array([int(ptr[0] + grid[i][0]), int(ptr[1] + grid[i][1]), int(ptr[2] + grid[i][2])], dtype=np.int32)
        # e_pt =  np.array([int(ptr_end[0] + grid[i][0]), int(ptr_end[1] + grid[i][1]), int(ptr_end[2] + grid[i][2])], dtype=np.int32)
        s_pt = tuple([int(ptr[0] + grid[i][0]), int(ptr[1] + grid[i][1]), int(ptr[2] + grid[i][2])] )
        e_pt =  tuple([int(ptr_end[0] + grid[i][0]), int(ptr_end[1] + grid[i][1]), int(ptr_end[2] + grid[i][2])] )
        # print(f"{s_pt} - {gm_map[s_pt]}")
        # print(f"{e_pt} - {gm_map[e_pt]}")
        # s_pt = [int(ptr[0]) + grid[i][0], int(ptr[1]) + grid[i][1], int(ptr[2]) + grid[i][2]] 
        # e_pt =  [int(ptr_end[0]) + grid[i][0], int(ptr_end[1]) + grid[i][1], int(ptr_end[2]) + grid[i][2]] 

        if gm_map[s_pt] > 0 and found1==0:
            roi1 = gm_map[s_pt]
            found1 += 1 

        if gm_map[e_pt] > 0 and found2==0:
            roi2 = gm_map[e_pt]
            found2 += 1
        
        if found1 + found2 == 2:
            break
    return [roi1, roi2]


def assign( input_tractogram: str, gm_map_file: str, out_assignment: str, threshold: double, verbose: bool=False, force: bool=False ):
    """Compute the assignments of the streamlines based on a GM map.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    out_assignment : string
        Path to the file (.txt or .npy) where to store the computed assignments.

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    ui.set_verbose( 2 if verbose else 1 )
    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if not os.path.isfile(gm_map_file):
        ui.ERROR( f'File "{gm_map_file}" not found' )


    # cdef int [:,:,::1] gm_map = np.ascontiguousarray(end_point_map, dtype = np.int32)
    # gm_map = np.ascontiguousarray(end_point_map, dtype = np.int32)

    out_assignment_ext = os.path.splitext(out_assignment)[1]
    out_assignment = f"{out_assignment[:-4]}_{threshold}{out_assignment_ext}"
    print(f"assign out: {out_assignment}")
    if out_assignment_ext not in ['.txt', '.npy']:
        ui.ERROR( 'Invalid extension for the output scalar file' )
    if os.path.isfile(out_assignment) and not force:
        ui.ERROR( 'Output scalar file already exists, use -f to overwrite' )
    
    gm_map_load = nib.load(gm_map_file).get_fdata()

    # threshold += 0.5
    gm_map = np.zeros((gm_map_load.shape[0] + 10, gm_map_load.shape[1] + 10, gm_map_load.shape[2] + 10))
    gm_map[:-10,:-10,:-10] = gm_map_load.astype(int)

    #----- iterate over input streamlines -----
    TCK_in = None
    
    thr = np.ceil(threshold).astype(np.int32)
    grid = compute_grid( thr )
    try:
        # open the input file
        TCK_in = LazyTCK( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        if verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )

        assignments = np.empty( (n_streamlines, 2), dtype=np.int32 )

        if n_streamlines>0:
            # for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            for i in xrange( n_streamlines):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                assignments[i] = streamline_assignment( TCK_in.streamline,  TCK_in.n_pts, grid, gm_map )
    except BaseException as e:
        if os.path.isfile( out_assignment ):
            os.remove( out_assignment )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if out_assignment_ext=='.txt':
            with open(out_assignment, "w") as text_file:
                for reg in assignments:
                    print(f"{int(reg[0])} {int(reg[1])}", file=text_file)
        else:
            np.save( out_assignment, assignments, allow_pickle=False )
        # if TCK_in is not None:
        #     TCK_in.close()