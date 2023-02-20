#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

import cython
import sys
import os 
import time

import numpy as np
cimport numpy as np
import nibabel as nib

from lazytractogram cimport LazyTractogram
from . import ui
from tqdm import tqdm
from libc.math cimport sqrt
from joblib import Parallel, delayed, cpu_count
from libc.math cimport floor




cdef compute_grid( float thr, float[:] vox_dim ) :

    """ Compute the offsets grid
    
        Parameters
        ---------------------
        thr : double
            Radius of the radial search
            
        vox_dim : 1x3 numpy array
            Voxel dimensions
    
    """

    cdef double grid_center[3]
    cdef int thr_grid = <int>thr
    cdef int m = 2
    
    # grid center
    cdef float x = vox_dim[0]/m
    cdef float y = vox_dim[1]/m
    cdef float z = vox_dim[2]/m
    
    grid_center[:] = [ x, y, z ]
    
    
    # create the mesh    
    mesh = np.linspace( -thr_grid, thr_grid, 2*thr_grid +1 )
    mx, my, mz = np.meshgrid( mesh, mesh, mesh )
    
    
    # find the centers of each voxels
    centers = np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1) + .5


    # sort the centers based on their distance from grid_center 
    cdef long[:] dist_grid = ((centers - grid_center)**2).sum(axis=1).argsort()
    cdef float[:,::1] centers_c = centers[ dist_grid ].astype(np.float32)

    return centers_c




cpdef to_matrix( float[ :,: ] streamline, int n ):

    """ Retrieve the coordinates of the streamlines' endpoints.
    
    Parameters
    -----------------
    streamline: Nx3 numpy array
        The streamline data
        
    n: int
        Writes first n points of the streamline. If n<0 (default), writes all points.

    """

    
    cdef float *ptr = &streamline[0,0]
    cdef float *ptr_end = ptr+n*3-3
        
    return np.array([ptr[0], ptr[1], ptr[2], ptr_end[0], ptr_end[1], ptr_end[2]], dtype=np.float32)




cpdef streamline_assignment( float [:] mat, float [:,::1] grid,
                            int[:,:,::1] gm_v, float thr, float[:] vox_dim,
                            float[:,::1] inverse, float[::1,:] small_view, float[:] val_view):

    """ Compute the label assigned to each streamline endpoint and then returns a list of connected regions.

    Parameters
    --------------
    mat: numpy array
        Contains the coordinates of the first point and last point of the streamline
    grid: numpy array
        A grid of offsets
    gm_v: numpy array 
        Reshape of the gm mask
    dim: numpy array
        Stores the gm dimensions needed to convert 3D coordinates to index positions
    thr: double
        Radius of the radial search
    flag: boolean
        Assume value "true" if thr = 0
    """

    cdef int roi1 = 0
    cdef int roi2 = 0
    cdef int found1 = 0
    cdef int found2 = 0
    cdef float dist_s = 0
    cdef float dist_e = 0
    cdef size_t i = 0
    cdef size_t yy = 0

    cdef float [:] starting_pt = np.asarray(mat[:3])
    cdef float [:] ending_pt = np.asarray(mat[3:])
    cdef float [:] pts_start = np.zeros(3, dtype=np.float32)
    cdef float [:] pts_end = np.zeros(3, dtype=np.float32)
    cdef int [:]  start_vox = np.zeros(3, dtype=np.int32)
    cdef int [:]  end_vox = np.zeros(3, dtype=np.int32)


    for yy in xrange(3):
        pts_start[yy] = ((starting_pt[0]*small_view[0,yy] + starting_pt[1]*small_view[1,yy] + starting_pt[2]*small_view[2,yy]) + val_view[yy]) 
        pts_start[yy] += (vox_dim[yy]/2)
        pts_start[yy] = floor(pts_start[yy])
        pts_end[yy] = ((ending_pt[0]*small_view[0,yy] + ending_pt[1]*small_view[1,yy] + ending_pt[2]*small_view[2,yy]) + val_view[yy]) 
        pts_end[yy] += (vox_dim[yy]/2)
        pts_end[yy] = floor(pts_end[yy])


    cdef int grid_size = grid.shape[0]

    
    for i in xrange(grid_size):
        
        # from 3D coordinates to index
        start_vox[0] = <int>(pts_start[0] + grid[i][0])
        start_vox[1] = <int>(pts_start[1] + grid[i][1])
        start_vox[2] = <int>(pts_start[2] + grid[i][2])
        end_vox[0] = <int>(pts_end[0] + grid[i][0])
        end_vox[1] = <int>(pts_end[1] + grid[i][1])
        end_vox[2] = <int>(pts_end[2] + grid[i][2])

        dist_s = ( pts_start[0] - start_vox[0] )**2 + ( pts_start[1] - start_vox[1] )**2 + ( pts_start[2] - start_vox[2] )**2 
        dist_e = ( pts_end[0] - end_vox[0] )**2 + ( pts_end[1] - end_vox[1] )**2 + ( pts_end[2] - end_vox[2] )**2 
        if gm_v[ start_vox[0], start_vox[1], start_vox[2]] > 0 and found1==0 and dist_s <= thr**2 :
            roi1 = <int>gm_v[ start_vox[0], start_vox[1], start_vox[2]]
            found1 = 1 
        
        if gm_v[ end_vox[0], end_vox[1], end_vox[2]  ] > 0 and found2==0 and dist_e <= thr**2 :    
            roi2 = <int>gm_v[ end_vox[0], end_vox[1], end_vox[2]  ]
            found2 = 1
    
        if found1 + found2 == 2:
            break
    # print((roi1, roi2))
    return np.array([roi1, roi2], dtype=np.int32)




def assign( input_tractogram: str, gm_map_file: str, out_assignment: str, threshold: 2, verbose: bool=False, force: bool=False ):

    """ Compute the assignments of the streamlines based on a GM map.
    
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


    out_assignment_ext = os.path.splitext(out_assignment)[1]
    out_assignment = f"{out_assignment[:-4]}_{threshold}{out_assignment_ext}"
    print(f"assign out: {out_assignment}")
    if out_assignment_ext not in ['.txt', '.npy']:
        ui.ERROR( 'Invalid extension for the output scalar file' )
    if os.path.isfile(out_assignment) and not force:
        ui.ERROR( 'Output scalar file already exists, use -f to overwrite' )
    
    # Load of the gm map
    gm_map_img = nib.load(gm_map_file)
    gm_map_data = gm_map_img.get_fdata()
    gm_header = gm_map_img.header
    affine = gm_map_img.affine.astype(np.float64)
    cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm_map_data, dtype=np.int32)
    # gm_map = np.zeros((gm_map_data.shape[0] + 5, gm_map_data.shape[1] + 5, gm_map_data.shape[2] + 5))
    # gm_map = gm_map_data.astype(np.int32)

    cdef float [:,::1] inverse = np.linalg.inv(affine).astype(np.float32) #inverse of affine
    cdef float [::1,:] small_view = inverse[:-1,:-1].T 
    cdef float [:] val_view = inverse[:-1,-1]
    cdef float [:] zooms = np.asarray( gm_header.get_zooms(), dtype = np.float32 )

    cdef float thr = np.ceil(threshold).astype(np.int32)
    cdef float [:,::1] grid
    cdef size_t i = 0   

    grid = compute_grid( thr, zooms )
    TCK_in = None
    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )
    cdef float [:] matrix = np.zeros( 6, dtype=np.float32)
    # cdef int [:,:] assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    try:      
        if verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )


        if n_streamlines>0:
            for i in xrange( n_streamlines ):
                print(f"streamline {i}", end="\r")
                TCK_in.read_streamline()

                # store the coordinates of the starting point and ending point
                matrix = to_matrix( TCK_in.streamline, int(TCK_in.n_pts) )
                assignments[i] = streamline_assignment( matrix, grid, gm_map, thr, zooms,  inverse, small_view, val_view)



    except BaseException as e:
        if os.path.isfile( out_assignment ):
            os.remove( out_assignment )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if out_assignment_ext=='.txt':
            with open(out_assignment, "w") as text_file:
                for reg in assignments:
                    print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
        else:
            np.save( out_assignment, assignments, allow_pickle=False )
        if TCK_in is not None:
            TCK_in.close()