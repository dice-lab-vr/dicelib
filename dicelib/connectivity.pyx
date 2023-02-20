#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

import cython
import sys
import os 
import time

import numpy as np
cimport numpy as np
import nibabel as nib
 
from .lazytck import LazyTCK
from . import ui
from tqdm import tqdm
from libc.math cimport sqrt
from joblib import Parallel, delayed, cpu_count
from libc.math cimport floor




cdef compute_grid( double thr, double[:] vox_dim ) :

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
    cdef double[:,:] centers_c = centers[ dist_grid ]

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
        
    return ptr[0], ptr[1], ptr[2], ptr_end[0], ptr_end[1], ptr_end[2]




cpdef streamline_assignment( float [:] mat, double [:,:] grid, int[:] gm_v, int[:] gm_dim, double thr, bint flag ) :

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

    
    cdef float *ptr = &mat[0]
    cdef int *gm_ptr = &gm_v[0]  
    
    cdef int roi1 = 0
    cdef int roi2 = 0
    cdef int found1 = 0
    cdef int found2 = 0
    cdef float dist_s
    cdef float dist_e
    cdef int s_idx
    cdef int e_idx
    cdef size_t i
    cdef int start_vox[3]
    cdef int end_vox[3]
    cdef int *ptr_start_vox = &start_vox[0]
    cdef int *ptr_end_vox = &end_vox[0]
    
    cdef int grid_size = grid.shape[0]
    
    
    for i in xrange(grid_size):
        
        # from 3D coordinates to index
        start_vox[:] = [ <int>(ptr[0] + grid[i][0]) , ( <int>(ptr[1] + grid[i][1]) ),  <int>(ptr[2] + grid[i][2]) ] 
        end_vox[:] = [ <int>(ptr[3] + grid[i][0]) , ( <int>(ptr[4] + grid[i][1]) ) , <int>(ptr[5] + grid[i][2]) ]


        s_idx = ptr_start_vox[0]*gm_dim[1]*gm_dim[2]  + ptr_start_vox[1]*gm_dim[2] + ptr_start_vox[2]
        e_idx = ptr_end_vox[0]*gm_dim[1]*gm_dim[2] + ptr_end_vox[1]*gm_dim[2] + ptr_end_vox[2]

        # if flag is set to True -> assignment_end_voxels
        if flag == True:
            roi1 = gm_ptr[s_idx]
            roi2 = gm_ptr[e_idx]
        
        # else -> assignment_radial_search
        else:
            dist_s = ( ptr[0] - ptr_start_vox[0] )**2 + ( ptr[1] - ptr_start_vox[1] )**2 + ( ptr[2] - ptr_start_vox[2] )**2 
            dist_e = ( ptr[3] - ptr_end_vox[0] )**2 + ( ptr[4] - ptr_end_vox[1] )**2 + ( ptr[5] - ptr_end_vox[2] )**2 
        
            if gm_ptr[ s_idx ] > 0 and found1==0 and dist_s <= thr**2 :
                roi1 = gm_ptr[s_idx]
                found1 += 1 
            
            if gm_ptr[ e_idx ] > 0 and found2==0 and dist_e <= thr**2 :    
                roi2 = gm_ptr[e_idx]
                found2 += 1
        
            if found1 + found2 == 2:
                break
        
    return [roi1, roi2]




def assign( input_tractogram: str, gm_map_file: str, out_assignment: str, threshold: double, verbose: bool=False, force: bool=False ):

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
     
    gm_map = np.zeros((gm_map_data.shape[0] + 5, gm_map_data.shape[1] + 5, gm_map_data.shape[2] + 5))
    gm_map[:-5,:-5,:-5] = gm_map_data.astype(int)
    
    # reshape of the gm 
    cdef int vals = <int>gm_map.shape[0] * <int>gm_map.shape[1] * <int>gm_map.shape[2]    
    gm_vec = np.reshape( gm_map, vals , order ='C' ).astype(np.int32)
    
    # retrieve dimensions
    gm_dim = np.asarray( gm_map.shape, dtype = np.int32 )
    zooms = np.asarray( gm_header.get_zooms(), dtype = np.float64 )


    #----- iterate over input streamlines -----
    
    
    cdef bint flag = False
    TCK_in = None
   
    # load of the .tck file
    TCK_in = LazyTCK( input_tractogram, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )
    
    assignments = np.empty( (n_streamlines, 2), dtype=np.int32 )
    matrix = np.empty( (n_streamlines, 6), dtype=np.float32 )

    thr = np.ceil(threshold).astype(np.int32)
    
    # set the flag value
    if thr == 0:
        flag = True
    else:
        flag = False
    
    
    # define the offsets grid
    grid = compute_grid( thr, zooms )
    
    
    # batch_size for joblib
    cdef int btch_s = int(n_streamlines/cpu_count()) + 1

    try:      
        if verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )


        if n_streamlines>0:
            for i in xrange( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                # store the coordinates of the starting point and ending point
                matrix[i] = to_matrix( TCK_in.streamline, TCK_in.n_pts )

        assignments = Parallel( n_jobs = -1, batch_size = btch_s, backend="threading" ) ( 
            delayed( streamline_assignment ) ( matrix[i], grid, gm_vec, gm_dim, thr, flag ) for i in xrange( n_streamlines ) )
        
            
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