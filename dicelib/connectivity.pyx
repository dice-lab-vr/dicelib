#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

import cython
import sys
import os 
import time

import numpy as np
cimport numpy as np
import nibabel as nib
# from nibabel.affines import apply_affine

from lazytractogram cimport LazyTractogram
from . import ui
from tqdm import tqdm
from libc.math cimport sqrt
from joblib import Parallel, delayed, cpu_count
from libc.math cimport floor
from libc.stdlib cimport malloc, free



cdef float [:,::1] apply_affine(float [:,::1] end_pts, float[:,::1] inverse, float[::1,:] M,
                                float[:] abc, float [:] vox_dim, float [:,::1] end_pts_trans) nogil:

    end_pts_trans[0][0] = ((end_pts[0][0]*M[0,0] + end_pts[0][1]*M[1,0] + end_pts[0][2]*M[2,0]) + abc[0]) + vox_dim[0]/2
    end_pts_trans[0][1] = ((end_pts[0][1]*M[0,1] + end_pts[1][1]*M[1,1] + end_pts[1][2]*M[2,1]) + abc[1]) + vox_dim[1]/2
    end_pts_trans[0][2] = (( end_pts[0][2]*M[0,2] + end_pts[0][1]*M[1,2] + end_pts[0][2]*M[2,2]) + abc[2]) + vox_dim[2]/2
    end_pts_trans[1][0] = ((end_pts[1][0]*M[0,0] + end_pts[1][1]*M[1,0] + end_pts[1][2]*M[2,0]) + abc[0]) + vox_dim[0]/2
    end_pts_trans[1][1] = (( end_pts[1][1]*M[0,1] + end_pts[0][1]*M[1,1] + end_pts[0][2]*M[2,1]) + abc[1]) + vox_dim[1]/2
    end_pts_trans[1][2] = ((end_pts[1][2]*M[0,2] + end_pts[1][1]*M[1,2] + end_pts[1][2]*M[2,2]) + abc[2]) + vox_dim[2]/2

    return end_pts_trans


cdef compute_grid( float thr, float[:] vox_dim ) :

    """ Compute the offsets grid
    
        Parameters
        ---------------------
        thr : double
            Radius of the radial search
            
        vox_dim : 1x3 numpy array
            Voxel dimensions
    
    """

    cdef float grid_center[3]
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




cpdef float [:,::1] to_matrix( float[:,::1] streamline, int n, float [:,::1] end_pts ) nogil:

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

    end_pts[0,0]=ptr[0]
    end_pts[0,1]=ptr[1]
    end_pts[0,2]=ptr[2]
    end_pts[1,0]=ptr_end[0]
    end_pts[1,1]=ptr_end[1]
    end_pts[1,2]=ptr_end[2]

    return end_pts




cdef int[:] streamline_assignment( int [:] start_vox, int [:] end_vox, int [:] roi_ret, float [:,::1] mat, float [:,::1] grid,
                            int[:,:,::1] gm_v, float thr) nogil:

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

    cdef float [:] starting_pt = mat[0]
    cdef float [:] ending_pt = mat[1]


    cdef int grid_size = grid.shape[0]
    for i in xrange(grid_size):
        # from 3D coordinates to index
        start_vox[0] = <int>(starting_pt[0] + grid[i][0])
        start_vox[1] = <int>(starting_pt[1] + grid[i][1])
        start_vox[2] = <int>(starting_pt[2] + grid[i][2])
        end_vox[0] = <int>(ending_pt[0] + grid[i][0])
        end_vox[1] = <int>(ending_pt[1] + grid[i][1])
        end_vox[2] = <int>(ending_pt[2] + grid[i][2])
        dist_s = sqrt( ( starting_pt[0] - start_vox[0] )**2 + ( starting_pt[1] - start_vox[1] )**2 + ( starting_pt[2] - start_vox[2] )**2 )
        dist_e = sqrt( ( ending_pt[0] - end_vox[0] )**2 + ( ending_pt[1] - end_vox[1] )**2 + ( ending_pt[2] - end_vox[2] )**2 )

        if gm_v[ start_vox[0], start_vox[1], start_vox[2]] > 0 and found1==0 and dist_s <= thr:
            roi_ret[0] = <int>gm_v[ start_vox[0], start_vox[1], start_vox[2]]
            found1 = 1
        
        if gm_v[ end_vox[0], end_vox[1], end_vox[2]  ] > 0 and found2==0 and dist_e <= thr:    
            roi_ret[1] = <int>gm_v[ end_vox[0], end_vox[1], end_vox[2]  ]
            found2 = 1
    
        if found1 and found2 :
            break
    return roi_ret




def assign( input_tractogram: str, start_chunk: int, end_chunk: int, chunk_size: int, reference: str,
            gm_map_file: str, out_assignment: str, threshold: 2, verbose: bool=False, force: bool=False ):

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

    
    # Load of the gm map
    gm_map_img = nib.load(gm_map_file)
    gm_map_data = gm_map_img.get_fdata()
    ref_data = nib.load(reference)
    ref_header = ref_data.header
    affine = ref_data.affine
    cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm_map_data, dtype=np.int32)
    # gm_map = np.zeros((gm_map_data.shape[0] + 5, gm_map_data.shape[1] + 5, gm_map_data.shape[2] + 5))
    # gm_map = gm_map_data.astype(np.int32)

    cdef float [:,::1] inverse = np.linalg.inv(affine).astype(np.float32) #inverse of affine
    cdef float [::1,:] M = inverse[:3, :3].T 
    cdef float [:] abc = inverse[:3, 3]
    cdef float [:] voxdims = np.asarray( ref_header.get_zooms(), dtype = np.float32 )

    cdef float thr = np.ceil(threshold).astype(np.float32)
    cdef float [:,::1] grid
    cdef size_t i = 0  
    cdef int start_i = 0  
    # cdef int n_streamlines = end_chunk
    cdef int n_streamlines = end_chunk - start_chunk
    cdef int start_c = <int> start_chunk

    
    grid = compute_grid( thr, voxdims )
    TCK_in = None
    TCK_in = LazyTractogram( input_tractogram, mode='r' )

    print('before while')
    print(f"start: {start_chunk}, end {end_chunk}")
    
    
    
    print('after while')


    cdef float [:,::1] matrix = np.zeros( (2,3), dtype=np.float32)
    # cdef int [:,:] assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    cdef int[:,:] assignments_view = assignments

    cdef float [:,::1] end_pts = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_temp = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_trans = np.zeros((2,3), dtype=np.float32)
    cdef int [:] start_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] end_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] roi_ret = np.array([0,0], dtype=np.int32)

    with nogil:
        while start_i < start_c:
            TCK_in._read_streamline()
            start_i += 1
        for i in xrange( n_streamlines ):
            # with gil:print(f"{i}/{n_streamlines}")
            # if i == 10:break
            TCK_in._read_streamline()
            # store the coordinates of the starting point and ending point
            end_pts = to_matrix( TCK_in.streamline, TCK_in.n_pts, end_pts_temp )
            # with gil:print(f"end_pts: {end_pts[0][0]} {end_pts[0][1]} {end_pts[0][2]}")
            # with gil:print(f"end_pts: {end_pts[1][0]} {end_pts[1][1]} {end_pts[1][2]}")
            matrix = apply_affine(end_pts, inverse, M, abc, voxdims, end_pts_trans)
            # with gil:print(f"matrix: {matrix[0][0]} {matrix[0][1]} {matrix[0][2]}")
            # with gil:print(f"matrix: {matrix[1][0]} {matrix[1][1]} {matrix[1][2]}")
            assignments_view[i] = streamline_assignment( start_vox, end_vox, roi_ret, matrix, grid, gm_map, thr)


    if TCK_in is not None:
        TCK_in.close()
    return assignments