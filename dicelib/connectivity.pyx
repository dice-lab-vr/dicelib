# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

from dicelib.streamline import create_replicas
from dicelib.ui import __logger__ as logger, ProgressBar, set_verbose
from dicelib.utils import check_params, File, Num

import os

import nibabel as nib
import numpy as np
from scipy.linalg import inv

from dicelib.streamline cimport apply_affine
from dicelib.tractogram cimport LazyTractogram

from libc.math cimport round as cround, sqrt
from libcpp cimport bool

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
    cdef int thr_grid = <int> np.ceil(thr)

    # grid center
    cdef float x = 0
    cdef float y = 0
    cdef float z = 0
    cdef float[:,::1] centers_c
    cdef int[:] dist_grid

    grid_center[:] = [ x, y, z ]

    # create the mesh    
    mesh = np.linspace( -thr_grid, thr_grid, 2*thr_grid +1 )
    mx, my, mz = np.meshgrid( mesh, mesh, mesh )

    # find the centers of each voxels
    centers = np.stack([mx.ravel() + x, my.ravel() + y, mz.ravel() + z], axis=1)

    # sort the centers based on their distance from grid_center
    dist_grid = ((centers - grid_center)**2).sum(axis=1).argsort().astype(np.int32)
    centers_c = centers[ dist_grid ].astype(np.float32)

    return centers_c




cpdef float [:,::1] to_matrix( float[:,::1] streamline, int n, float [:,::1] end_pts ) noexcept nogil:

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


cdef float distance2vox(float vox_x_min, float vox_x_max, float vox_y_min, float vox_y_max, float vox_z_min, float vox_z_max, float p_x, float p_y, float p_z) nogil:
    cdef float dx = max(vox_x_min - p_x, 0, p_x - vox_x_max)
    cdef float dy = max(vox_y_min - p_y, 0, p_y - vox_y_max)
    cdef float dz = max(vox_z_min - p_z, 0, p_z - vox_z_max)
    return sqrt(dx*dx + dy*dy + dz*dz)


cdef int[:] streamline_assignment_endpoints( int[:] start_vox, int[:] end_vox, int [:] roi_ret, float [:,::1] mat, int[:,:,::1] gm_v) noexcept nogil:

    cdef float [:] starting_pt = mat[0]
    cdef float [:] ending_pt = mat[1]
    start_vox[0] = <int> starting_pt[0]
    start_vox[1] = <int> starting_pt[1]
    start_vox[2] = <int> starting_pt[2]
    end_vox[0]   = <int> ending_pt[0]
    end_vox[1]   = <int> ending_pt[1]
    end_vox[2]   = <int> ending_pt[2]

    roi_ret[0] = gm_v[ start_vox[0], start_vox[1], start_vox[2]]
    roi_ret[1] = gm_v[ end_vox[0], end_vox[1], end_vox[2]]
    return roi_ret


cdef int[:] streamline_assignment( float [:] start_pt_grid, int[:] start_vox, float [:] end_pt_grid, int[:] end_vox, int [:] roi_ret, float [:,::1] mat, float [:,::1] grid,
                            int[:,:,::1] gm_v, float thr, int[:] count_neighbours) noexcept nogil:

    """ Compute the label assigned to each streamline endpoint and then returns a list of connected regions.

    Parameters
    --------------
    start_pt_grid : 1x3 numpy array
        Starting point of the streamline in the grid space.
    start_vox : 1x3 numpy array
        Starting point of the streamline in the voxel space.
    end_pt_grid : 1x3 numpy array
        Ending point of the streamline in the grid space.
    end_vox : 1x3 numpy array
        Ending point of the streamline in the voxel space.
    roi_ret : 1x2 numpy array
        Labels assigned to the streamline endpoints.
    mat : 2x3 numpy array
        Streamline endpoints.
    grid : Nx3 numpy array
        Grid of voxels to check.
    gm_v : 3D numpy array
        GM map.
    thr : float
        Threshold used to compute the grid of voxels to check.
    """

    cdef float dist_s = 0
    cdef float dist_e = 0
    cdef size_t i = 0
    cdef int idx_s_min = 0
    cdef int idx_e_min = 0
    cdef float dist_s_temp = 1000
    cdef float dist_e_temp = 1000
    cdef int layer = 0

    roi_ret[0] = 0
    roi_ret[1] = 0

    cdef float [:] starting_pt = mat[0]
    cdef float [:] ending_pt = mat[1]
    cdef int grid_size = grid.shape[0]

    cdef float vox_x_min = 0
    cdef float vox_x_max = 0
    cdef float vox_y_min = 0
    cdef float vox_y_max = 0
    cdef float vox_z_min = 0
    cdef float vox_z_max = 0

    for i in xrange(grid_size):

        # from 3D coordinates to index
        start_pt_grid[0] = starting_pt[0] + grid[i][0]
        start_pt_grid[1] = starting_pt[1] + grid[i][1]
        start_pt_grid[2] = starting_pt[2] + grid[i][2]

        # check if the voxel is inside the mask
        if start_pt_grid[0] < 0 or start_pt_grid[0] >= gm_v.shape[0] or start_pt_grid[1] < 0 or start_pt_grid[1] >= gm_v.shape[1] or start_pt_grid[2] < 0 or start_pt_grid[2] >= gm_v.shape[2]:
            continue

        start_vox[0] = <int> start_pt_grid[0]
        start_vox[1] = <int> start_pt_grid[1]
        start_vox[2] = <int> start_pt_grid[2]

        if gm_v[ start_vox[0], start_vox[1], start_vox[2] ] > 0:
            vox_x_min = <int>(start_pt_grid[0])
            vox_x_max = <int>(start_pt_grid[0]) + 1
            vox_y_min = <int>(start_pt_grid[1])
            vox_y_max = <int>(start_pt_grid[1]) + 1
            vox_z_min = <int>(start_pt_grid[2])
            vox_z_max = <int>(start_pt_grid[2]) + 1
            dist_s = distance2vox(vox_x_min, vox_x_max, vox_y_min, vox_y_max, vox_z_min, vox_z_max, starting_pt[0], starting_pt[1], starting_pt[2])

            if dist_s <= thr and dist_s < dist_s_temp:
                roi_ret[0] = gm_v[ start_vox[0], start_vox[1], start_vox[2]]
                dist_s_temp = dist_s
        if i == count_neighbours[layer]:
            if dist_s_temp < 1000:
                break
            else:
                layer += 1
    layer = 0
    for i in xrange(grid_size):
        end_pt_grid[0] = ending_pt[0] + grid[i][0]
        end_pt_grid[1] = ending_pt[1] + grid[i][1]
        end_pt_grid[2] = ending_pt[2] + grid[i][2]

        if end_pt_grid[0] < 0 or end_pt_grid[0] >= gm_v.shape[0] or end_pt_grid[1] < 0 or end_pt_grid[1] >= gm_v.shape[1] or end_pt_grid[2] < 0 or end_pt_grid[2] >= gm_v.shape[2]:
            continue

        end_vox[0] = <int> end_pt_grid[0]
        end_vox[1] = <int> end_pt_grid[1]
        end_vox[2] = <int> end_pt_grid[2]

        if gm_v[ end_vox[0], end_vox[1], end_vox[2] ] > 0:
            vox_x_min = <int>(end_pt_grid[0])
            vox_x_max = <int>(end_pt_grid[0]) + 1
            vox_y_min = <int>(end_pt_grid[1])
            vox_y_max = <int>(end_pt_grid[1]) + 1
            vox_z_min = <int>(end_pt_grid[2])
            vox_z_max = <int>(end_pt_grid[2]) + 1
            dist_e = distance2vox(vox_x_min, vox_x_max, vox_y_min, vox_y_max, vox_z_min, vox_z_max, ending_pt[0], ending_pt[1], ending_pt[2])

            if dist_e <= thr and dist_e < dist_e_temp:
                roi_ret[1] = gm_v[ end_vox[0], end_vox[1], end_vox[2]]
                dist_e_temp = dist_e
        if i == count_neighbours[layer]:
            if dist_e_temp < 1000:
                break
            else:
                layer += 1

    return roi_ret


cpdef assign( input_tractogram: str, int[:] pbar_array, int id_chunk, int start_chunk, int end_chunk, gm_map_file: str, threshold: 2 ):

    """ Compute the assignments of the streamlines based on a GM atlas.
    
    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.
    pbar_array : numpy array
        Array of integers used to update the progress bar.
    id_chunk : int
        Index of the chunk.
    start_chunk : int
        Index of the first streamline of the chunk.
    end_chunk : int
        Index of the last streamline of the chunk.
    gm_map_file : string
        Path to the GM map file.
    threshold : int
        Threshold used to compute the grid of voxels to check.
    """


    if not os.path.isfile(input_tractogram):
        logger.error( f'File "{input_tractogram}" not found' )
    if not os.path.isfile(gm_map_file):
        logger.error( f'File "{gm_map_file}" not found' )

    
    # Load of the gm map
    gm_map_img = nib.load(gm_map_file)
    gm_map_data = gm_map_img.get_fdata()
    ref_data = gm_map_img
    ref_header = ref_data.header
    affine = ref_data.affine
    cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm_map_data, dtype=np.int32)

    cdef float [:,::1] inverse = np.ascontiguousarray(inv(affine), dtype=np.float32) #inverse of affine
    cdef float [::1,:] M = inverse[:3, :3].T 
    cdef float [:] abc = inverse[:3, 3]
    cdef float [:] voxdims = np.asarray( ref_header.get_zooms(), dtype = np.float32 )

    cdef float thr = <float> threshold/np.max(voxdims)
    cdef float [:,::1] grid
    cdef size_t i = 0  
    cdef int n_streamlines = end_chunk - start_chunk
    cdef float [:,::1] matrix = np.zeros( (2,3), dtype=np.float32)
    assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    cdef int[:,:] assignments_view = assignments

    cdef float [:,::1] end_pts = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_temp = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_trans = np.zeros((2,3), dtype=np.float32)
    cdef float [:] start_pt_grid = np.zeros(3, dtype=np.float32)
    cdef int [:] start_vox = np.zeros(3, dtype=np.int32)
    cdef float [:] end_pt_grid = np.zeros(3, dtype=np.float32)
    cdef int [:] end_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] roi_ret = np.array([0,0], dtype=np.int32)

    TCK_in = None
    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    # compute the grid of voxels to check
    grid = compute_grid( thr, voxdims )
    layers = np.arange( 0,<int> np.ceil(thr)+1, 1 ) # e.g [0, 1, 2, 3]
    lato = layers * 2 + 1 # e.g [0, 3, 5, 7] = layerx2+1
    neighbs = [v**3-1 for v in lato] # e.g [1, 27, 125, 343] = (lato)**3
    cdef int[:] count_neighbours = np.array(neighbs, dtype=np.int32)

    if thr < 0.5 :
        with nogil:
            while i < start_chunk:
                TCK_in._read_streamline()
                i += 1
            for i in xrange( n_streamlines ):
                TCK_in._read_streamline()
                end_pts = to_matrix( TCK_in.streamline, TCK_in.n_pts, end_pts_temp )
                matrix = apply_affine(end_pts, M, abc, end_pts_trans)
                assignments_view[i] = streamline_assignment_endpoints( start_vox, end_vox, roi_ret, matrix, gm_map)
                pbar_array[id_chunk] += 1

    else:
        with nogil:
            while i < start_chunk:
                TCK_in._read_streamline()
                i += 1
            for i in xrange( n_streamlines ):
                TCK_in._read_streamline()
                end_pts = to_matrix( TCK_in.streamline, TCK_in.n_pts, end_pts_temp )
                matrix = apply_affine(end_pts, M, abc, end_pts_trans)
                assignments_view[i] = streamline_assignment( start_pt_grid, start_vox, end_pt_grid, end_vox, roi_ret,
                                                            matrix, grid, gm_map, thr, count_neighbours)
                pbar_array[id_chunk] += 1

    if TCK_in is not None:
        TCK_in.close()
    return assignments


def compute_connectome_blur( input_tractogram: str, output_connectome: str, weights_in: str, input_nodes: str,
                            blur_core_extent: float, blur_gauss_extent: float, blur_spacing: float=0.25,
                            blur_gauss_min: float=0.1, offset_thr: float=0.0, symmetric: bool=False, fiber_shift=0,
                            verbose: int=3, force: bool=False ):
    """Build the connectome weighted by COMMITblur (only sum).

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_connectome : string
        Path to the file where to store the resulting connectome.

    weights_in : string
        Scalar file (.txt or .npy) for the input streamline weights estimated by COMMITblur.

    input_nodes : string
        Path to the file containing the gray matter parcellation (nodes of the connectome).

    blur_core_extent: float
        Extent of the core inside which the segments have equal contribution to the central one used by COMMITblur.

    blur_gauss_extent: float
        Extent of the gaussian damping at the border used by COMMITblur.

    blur_spacing : float
        To obtain the blur effect, streamlines are duplicated and organized in a cartesian grid;
        this parameter controls the spacing of the grid in mm (defaut : 0.25).

    blur_gauss_min: float
        Minimum value of the Gaussian to consider when computing the sigma (default : 0.1).

    offset_thr: float
        Quantity added to the threshold used to compute the assignments of the replicas. 
        If the input streamlines don't have both ending points inside a GM region, increase this value (default : 0.0).

    symmetric : boolean
        Make output connectome symmetric (default : False).

    fiber_shift : float or list of three float
        If necessary, apply a translation to streamline coordinates (default : 0) to account
        for differences between the reference system of the tracking algorithm and COMMIT.
        The value is specified in voxel units, eg 0.5 translates by half voxel.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose(verbose)

    logger.info( 'Compute connectome weighted by COMMITblur' )    

    # check input tractogram
    if not os.path.isfile(input_tractogram):
        logger.error( f'File "{input_tractogram}" not found' )
    logger.subinfo( f'Input tractogram: "{input_tractogram}"', indent_char='*')

    # output
    if os.path.isfile(output_connectome) and not force:
        logger.error( 'Output connectome already exists, use -f to overwrite' )
    conn_out_ext = os.path.splitext(output_connectome)[1]
    if conn_out_ext not in ['.csv', '.npy']:
        logger.error('Invalid extension for the output connectome file')

    # streamline weights
    if not os.path.isfile( weights_in ):
        logger.error( f'File "{weights_in}" not found' )
    weights_in_ext = os.path.splitext(weights_in)[1]
    if weights_in_ext=='.txt':
        w = np.loadtxt( weights_in ).astype(np.float64)
    elif weights_in_ext=='.npy':
        w = np.load( weights_in, allow_pickle=False ).astype(np.float64)
    else:
        logger.error( 'Invalid extension for the weights file' )

    # parcellation
    if not os.path.isfile(input_nodes):
        logger.error( f'File "{input_nodes}" not found' )
    logger.subinfo( f'Input parcellation: "{input_nodes}"', indent_char='*')

    # blur parameters
    if blur_core_extent<0:
        logger.error( '"blur_core_extent" must be >= 0' )
    if blur_gauss_extent<0:
        logger.error( '"blur_gauss_extent" must be >= 0' )
    if blur_spacing<=0:
        logger.error( '"blur_spacing" must be > 0' )
    if blur_gauss_min<=0:
        logger.error( '"blur_gauss_min" must be > 0' )
    logger.subinfo( 'Blur parameters:', indent_char='*')
    logger.subinfo( f'- blur_core_extent:  {blur_core_extent}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'- blur_gauss_extent: {blur_gauss_extent}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'- blur_spacing:      {blur_spacing}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'- blur_gauss_min:    {blur_gauss_min}', indent_lvl=1, indent_char='-')

    # fiber_shift
    if np.isscalar(fiber_shift) :
        fiber_shiftX = fiber_shift
        fiber_shiftY = fiber_shift
        fiber_shiftZ = fiber_shift
    elif len(fiber_shift) == 3 :
        fiber_shiftX = fiber_shift[0]
        fiber_shiftY = fiber_shift[1]
        fiber_shiftZ = fiber_shift[2]
    else :
        logger.error( '"fiber_shift" must be a scalar or a vector with 3 elements' )

    # load parcellation
    gm_nii = nib.load(input_nodes)
    gm = gm_nii.get_fdata()
    gm_header = gm_nii.header
    affine = gm_nii.affine
    cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm, dtype=np.int32)
    cdef float [:,::1] inverse = np.ascontiguousarray(inv(affine), dtype=np.float32) #inverse of affine
    cdef float [::1,:] M = inverse[:3, :3].T 
    cdef float [:] abc = inverse[:3, 3]
    cdef float [:] voxdims = np.asarray( gm_header.get_zooms(), dtype = np.float32 )

    # divide blur parameters by voxelsize bacause we use them in VOX space
    core_extent  = blur_core_extent/np.max(voxdims)
    gauss_extent = blur_gauss_extent/np.max(voxdims)
    spacing      = blur_spacing/np.max(voxdims)

    # blur parameters (like in trk2dictionary)
    cdef double [:] blurRho
    cdef double [:] blurAngle
    cdef double [:] blurWeights
    cdef int nReplicas
    cdef float blur_sigma
    # compute replicas coordinates
    tmp = np.arange(0,core_extent+gauss_extent+1e-6,spacing)
    tmp = np.concatenate( (tmp,-tmp[1:][::-1]) )
    x, y = np.meshgrid( tmp, tmp )
    r = np.sqrt( x*x + y*y )
    idx = (r <= core_extent+gauss_extent)
    blurRho = r[idx]
    blurAngle = np.arctan2(y,x)[idx]
    nReplicas = blurRho.size
    # compute replicas scaling factors
    blurWeights = np.empty( nReplicas, np.double  )
    if gauss_extent == 0 :
        blurWeights[:] = 1.0
    else:
        blur_sigma = gauss_extent / np.sqrt( -2.0 * np.log( blur_gauss_min ) )
        for i_r in xrange(nReplicas):
            if blurRho[i_r] <= core_extent :
                blurWeights[i_r] = 1.0
            else:
                blurWeights[i_r] = np.exp( -(blurRho[i_r] - core_extent)**2 / (2.0*blur_sigma**2) )
    logger.subinfo(f'Number of replicas for each streamline = {nReplicas}', indent_lvl=1, indent_char='-')

    # compute the grid of voxels for the radial search
    threshold = core_extent + gauss_extent
    # print(f'thr = {thr}')
    cdef float thr = threshold + (offset_thr/np.max(voxdims)) # if input streamlines are all connecting but using a radial search
    grid = compute_grid( thr, voxdims )
    layers = np.arange( 0,<int> np.ceil(thr)+1, 1 ) # e.g. layer=[0, 1, 2, 3]
    lato = layers * 2 + 1 # e.g. lato = [0, 3, 5, 7] = layerx2+1
    neighbs = [v**3-1 for v in lato] # e.g. [1, 27, 125, 343] = (lato)**3
    cdef int[:] count_neighbours = np.array(neighbs, dtype=np.int32)
    thr += 0.005 # to take into accound rounding errors in the distance of the replicas
    # print(f'core+gauss = {core_extent + gauss_extent}')
    logger.subinfo(f'Threshold to use when computing assignments (in VOX space) = {thr:.3f}', indent_lvl=1, indent_char='-')

    # variables for transformations 
    cdef float [:,::1] pts_start = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] pts_end   = np.zeros((2,3), dtype=np.float32)
    cdef float *ptr
    cdef float *ptr_end
    cdef float [:,::1] pts_start_tmp = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] pts_end_tmp   = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] pts_start_vox = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] pts_end_vox   = np.zeros((2,3), dtype=np.float32)

    # variables for replicas creation
    cdef float [:,::1] replicas_start = np.zeros((3,nReplicas), dtype=np.float32)
    cdef float [:,::1] replicas_end   = np.zeros((nReplicas,3), dtype=np.float32)
    cdef double [:] blurWeights_norm  = blurWeights/np.sum(blurWeights) # normalize in order to have sum = 1

    # variables for assignments
    asgn = np.zeros( (nReplicas, 2), dtype=np.int32 )
    cdef int[:,:] asgn_view = asgn
    cdef float [:] start_pt_grid = np.zeros(3, dtype=np.float32)
    cdef float [:] end_pt_grid   = np.zeros(3, dtype=np.float32)
    cdef int [:] start_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] end_vox   = np.zeros(3, dtype=np.int32)
    cdef int [:] roi_ret   = np.array([0,0], dtype=np.int32)
    cdef float [:,::1] points_mat = np.zeros( (2,3), dtype=np.float32)

    # create connectome to fill
    n_rois = np.max(gm).astype(np.int32)
    conn = np.zeros((n_rois, n_rois), dtype=np.float64)

    #----- iterate over input files -----
    TCK_in = None
    # TCK_out = None
    cdef size_t i, j, k = 0  
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        # output_tractogram = input_tractogram[:-4]+'_non_connecting.tck'
        # TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        # str_count = 0

        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo( f'{n_streamlines} streamlines in input tractogram', indent_char='*')

        # check if #(weights)==n_streamlines
        if n_streamlines!=w.size:
            logger.error( f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) ' )

        zeros_count = 0

        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                if w[i]>0:
                    # retrieve the coordinates of 2 points at each end
                    ptr = &TCK_in.streamline[0,0]
                    #first
                    pts_start[0,0]=ptr[0]
                    pts_start[0,1]=ptr[1]
                    pts_start[0,2]=ptr[2]
                    # second
                    pts_start[1,0]=ptr[3]
                    pts_start[1,1]=ptr[4]
                    pts_start[1,2]=ptr[5]

                    ptr_end = ptr+TCK_in.n_pts*3-3*2
                    # second-to-last
                    pts_end[1,0]=ptr_end[0]
                    pts_end[1,1]=ptr_end[1]
                    pts_end[1,2]=ptr_end[2]
                    # last
                    pts_end[0,0]=ptr_end[3]
                    pts_end[0,1]=ptr_end[4]
                    pts_end[0,2]=ptr_end[5]

                    # change space to VOX
                    pts_start_vox = apply_affine(pts_start, M, abc, pts_start_tmp) # starting points in voxel space
                    pts_end_vox   = apply_affine(pts_end,   M, abc, pts_end_tmp)   # ending points in voxel space

                    # create replicas of starting and ending points
                    replicas_start = create_replicas(pts_start_vox, blurRho, blurAngle, nReplicas, fiber_shiftX, fiber_shiftY, fiber_shiftZ)
                    replicas_end   = create_replicas(pts_end_vox,   blurRho, blurAngle, nReplicas, fiber_shiftX, fiber_shiftY, fiber_shiftZ)

                    # compute assignments of the replicas
                    for j in range(nReplicas):
                        points_mat = np.array([[replicas_start[j][0], replicas_start[j][1], replicas_start[j][2]], 
                                                [replicas_end[j][0], replicas_end[j][1], replicas_end[j][2]]],
                                                dtype=np.float32)
                        asgn_view[j][:] = streamline_assignment( start_pt_grid, start_vox, end_pt_grid, end_vox, roi_ret, points_mat, grid, gm_map, thr, count_neighbours)
                        
                    zeros_count += (asgn.size - np.count_nonzero(asgn))

                    # find unique assignments and sum the weights of their replicas
                    asgn_sort = np.sort(asgn, axis=1) # shape = (nReplicas, 2)
                    asgn_unique = np.unique(asgn_sort, axis=0) 
                    weight_fraction = np.zeros(asgn_unique.shape[0], dtype=np.float64) # one value for each unique pair of ROI
                    for j in range(nReplicas):
                        idx = np.where(np.all(asgn_unique==asgn_sort[j],axis=1)) # find idx in weight_fraction corresponding to the pair of ROI of the current replica
                        weight_fraction[idx] += blurWeights_norm[j] # total fraction of the blurred streamline weight to be assigned to a specific pair of ROI

                    # update the connectome weights
                    weight_fraction = np.round(weight_fraction * w[i], 12)
                    for k in range(asgn_unique.shape[0]):
                        if asgn_unique[k][0] == 0: continue
                        conn[asgn_unique[k][0]-1, asgn_unique[k][1]-1] += weight_fraction[k]

                pbar.update()

        if zeros_count > 0 : logger.warning(f'Some replicas are not assigned to any region (tot. {zeros_count})')


    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        # if TCK_out is not None:
        #     TCK_out.close( write_eof=True, count=str_count )
        if symmetric:
            conn_sym = conn.T + conn
            np.fill_diagonal(conn_sym,np.diag(conn))
            if conn_out_ext=='.csv':
                np.savetxt(output_connectome, conn_sym, delimiter=",")
            else:
                np.save(output_connectome, conn_sym, allow_pickle=False)
        else:
            if conn_out_ext=='.csv':
                np.savetxt(output_connectome, conn, delimiter=",")
            else:
                np.save(output_connectome, conn, allow_pickle=False)
    logger.subinfo( f'Writing output connectome to "{output_connectome}"', indent_char='*')



def build_connectome( input_assignments: str, output_connectome: str, input_weights: str=None, input_tractogram: str=None, input_nodes: str=None, atlas_dist: float=2.0, metric: str='sum', symmetric: bool=False, verbose: int=3, force: bool=False ):
    """Build the (weighted) connectome having the assignments or the tractogram and an atlas.

    Parameters
    ----------
    input_weights : string
        Scalar file (.txt or .npy) for the input streamline weights.
        
    input_assignments : string
        Path to the file (.txt or .npy) containing the streamline assignments.

    output_connectome : string
        Path to the file where to store the resulting connectome.

    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    input_nodes : string
        Path to the file containing the gray matter parcellation (nodes of the connectome).

    atlas_dist : float
        Distance [in mm] used to assign streamlines to the atlas' nodes (default: 2.0).

    metric : string
        Operation to compute the value of the edges, options: sum, mean, min, max (default: sum).

    symmetric : boolean
        Make output connectome symmetric (default : False).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose(verbose)
    logger.info('Computing connectome')

    files = [
        File(name='connectome_out', type_='output', path=output_connectome, ext=['.csv', '.npy'])
    ]
    if input_weights is not None:
        files.append(File(name='weights_in', type_='input', path=input_weights, ext=['.txt', '.npy']))
    if os.path.isfile(input_assignments):
        files.append(File(name='assignments_in', type_='input', path=input_assignments, ext=['.txt', '.npy']))
    else:
        # check input tractogram and parcellation
        if input_tractogram is None:
            logger.error(f'Tractogram file \'{input_tractogram}\' not found. Required if the assignments does not exist.')
        if input_nodes is None:
            logger.error(f'Nodes file \'{input_nodes}\' not found. Required if the assignments does not exist.')
        files.extend([
            File(name='tractogram_in', type_='input', path=input_tractogram, ext=['.tck']),
            File(name='nodes_in', type_='input', path=input_nodes, ext=['.nii', '.nii.gz'])
        ])

        # logger.info('No assignments file found. Computing assignments')
        logger.subinfo(f'Input tractogram: \'{input_tractogram}\'', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Input parcellation: \'{input_nodes}\'', indent_char='*', indent_lvl=1)

        # compute assignments
        logger.subinfo('Computing assignments', indent_lvl=1, indent_char='*', with_progress=verbose>2)
        with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=True) as pbar:
            os.system(f'dice_tractogram_assign {input_tractogram} {input_nodes} {input_assignments} -d {atlas_dist} -v 1')

    check_params(files=files, force=force)

    # streamline assignments
    input_assignments_ext = os.path.splitext(input_assignments)[1]
    if input_assignments_ext=='.txt':
        asgn = np.loadtxt( input_assignments ).astype(np.int32)
    else:
        asgn = np.load( input_assignments, allow_pickle=False ).astype(np.int32)
    n_streamlines = asgn.shape[0]
    asgn_sort = np.sort(asgn, axis=1) # shape = (n_streamlines, 2)
    
    # streamline weights
    if input_weights is None:
        w = np.ones( n_streamlines, dtype=np.int32 )
    else:
        input_weights_ext = os.path.splitext(input_weights)[1]
        if input_weights_ext=='.txt':
            w = np.loadtxt( input_weights ).astype(np.float64)
        elif input_weights_ext=='.npy':
            w = np.load( input_weights, allow_pickle=False ).astype(np.float64)
        # check if #(weights)==n_streamlines
        if n_streamlines != w.size:
            logger.error(f'# of weights ({w.size}) is different from # of streamline assignments ({n_streamlines})')

    # metric
    if metric not in ['sum', 'mean', 'min', 'max']:
        logger.error('Invalid type of metric for the edges. Options: sum, mean, min, max.')
    if input_weights is None:
        metric = 'sum' # to compute connectome NOS


    logger.subinfo(f'Streamline assignments: "{input_assignments}"', indent_char='*', indent_lvl=1)
    if input_weights is not None:
        logger.subinfo(f'Chosen metric to weight the edges: {metric}', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Input weights: "{input_weights}"', indent_char='*', indent_lvl=1)
    else:
        logger.subinfo('No weights provided, the connectome will contain the number of streamlines', indent_char='*', indent_lvl=1)

    # create connectome to fill
    if input_nodes is not None:
        gm_nii = nib.load(input_nodes)
        gm = gm_nii.get_fdata()
        n_rois = np.max(gm).astype(np.int32)
        logger.subinfo(f'Number of regions: {n_rois}', indent_char='*', indent_lvl=1)
    else:
        n_rois = np.max(asgn).astype(np.int32)
        logger.subinfo(f'Number of regions: {n_rois}', indent_char='*', indent_lvl=1)

    if metric == 'min':
        conn = np.triu(np.full((n_rois, n_rois), 1000000000, dtype=np.float64))
    elif metric == 'max':
        conn = np.triu(np.full((n_rois, n_rois), -1000000000, dtype=np.float64))
    else:
        conn = np.zeros((n_rois, n_rois), dtype=np.float64)
    conn_nos = np.zeros((n_rois, n_rois), dtype=np.float64)
    count_unconn = 0

    logger.subinfo('Building connectome', indent_char='*', indent_lvl=1)
    with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True, subinfo=False) as pbar:
        for i in range( n_streamlines ):
            if asgn_sort[i][0] == 0 or asgn_sort[i][1] == 0: 
                count_unconn += 1
                continue

            if metric == 'min':
                if w[i] < conn[asgn_sort[i][0]-1, asgn_sort[i][1]-1]:
                    conn[asgn_sort[i][0]-1, asgn_sort[i][1]-1] = w[i]
            elif metric == 'max':
                if w[i] > conn[asgn_sort[i][0]-1, asgn_sort[i][1]-1]:
                    conn[asgn_sort[i][0]-1, asgn_sort[i][1]-1] = w[i]
            else: # sum or mean
                conn[asgn_sort[i][0]-1, asgn_sort[i][1]-1] += w[i]
            conn_nos[asgn_sort[i][0]-1, asgn_sort[i][1]-1] += 1
        pbar.update()
    if count_unconn > 0:
        logger.warning(f'Number of non-connecting streamlines {count_unconn}')

    if metric == 'mean':
        conn[conn_nos>0] = conn[conn_nos>0]/conn_nos[conn_nos>0]

    conn[conn_nos==0] = 0
    # np.save(output_connectome[:-4]+'NOS.npy', conn_nos, allow_pickle=False)

    conn_out_ext = os.path.splitext(output_connectome)[1]
    if symmetric:
        conn_sym = conn.T + conn
        np.fill_diagonal(conn_sym,np.diag(conn))
        if conn_out_ext=='.csv':
            np.savetxt(output_connectome, conn_sym, delimiter=",")
        else:
            np.save(output_connectome, conn_sym, allow_pickle=False)
    else:
        if conn_out_ext=='.csv':
            np.savetxt(output_connectome, conn, delimiter=",")
        else:
            np.save(output_connectome, conn, allow_pickle=False)

    logger.subinfo( f'Writing output connectome to "{output_connectome}"', indent_char='*', indent_lvl=1)