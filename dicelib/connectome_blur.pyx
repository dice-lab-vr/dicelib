#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

import os 

import numpy as np
# cimport numpy as np
import nibabel as nib
# from nibabel.affines import apply_affine
from scipy.linalg import inv
# from libc.math cimport sqrt
# from libc.math cimport round as cround

from dicelib.lazytractogram cimport LazyTractogram
from . import ui
from dicelib.connectivity cimport apply_affine
from dicelib.connectivity cimport compute_grid
from dicelib.connectivity cimport streamline_assignment
from dicelib.streamline import create_replicas


def compute_connectome_blur( input_tractogram: str, output_connectome: str, weights_in: str, parcellation_in: str, blur_core_extent: float, blur_gauss_extent: float, blur_spacing: float=0.25, blur_gauss_min: float=0.1, offset_thr: float=0.0, symmetric: bool=False, fiber_shift=0, verbose: int=1, force: bool=False ):
    """Build the connectome weighted by COMMITblur.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_connectome : string
        Path to the file where to store the resulting connectome.

    weights_in : str
        Scalar file (.txt or .npy) for the input streamline weights estimated by COMMITblur.

    parcellation_in : str
        Path to the file containing the gray matter parcellation.

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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 1).

    force : boolean
        Force overwriting of the output (default : False).
    """

    ui.set_verbose( verbose )

    # check input 
    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if os.path.isfile(output_connectome) and not force:
        ui.ERROR( 'Output connectome already exists, use -f to overwrite' )
    conn_out_ext = os.path.splitext(output_connectome)[1]
    ui.INFO( f'Input tractogram: "{input_tractogram}"' )

    #streamline weights
    if not os.path.isfile( weights_in ):
        ui.ERROR( f'File "{weights_in}" not found' )
    weights_in_ext = os.path.splitext(weights_in)[1]
    if weights_in_ext=='.txt':
        w = np.loadtxt( weights_in ).astype(np.float64)
    elif weights_in_ext=='.npy':
        w = np.load( weights_in, allow_pickle=False ).astype(np.float64)
    else:
        ui.ERROR( 'Invalid extension for the weights file' )

    # parcellation
    if not os.path.isfile(parcellation_in):
        ui.ERROR( f'File "{parcellation_in}" not found' )
    ui.INFO( f'Input parcellation: "{parcellation_in}"' )

    # blur parameters
    if blur_core_extent<0:
        ui.ERROR( '"blur_core_extent" must be >= 0' )
    if blur_gauss_extent<0:
        ui.ERROR( '"blur_gauss_extent" must be >= 0' )
    if blur_spacing<=0:
        ui.ERROR( '"blur_spacing" must be > 0' )
    if blur_gauss_min<=0:
        ui.ERROR( '"blur_gauss_min" must be > 0' )
    ui.INFO( 'Blur parameters:')
    ui.INFO( f'- blur_core_extent:  {blur_core_extent}' )
    ui.INFO( f'- blur_gauss_extent: {blur_gauss_extent}' )
    ui.INFO( f'- blur_spacing:      {blur_spacing}' )
    ui.INFO( f'- blur_gauss_min:    {blur_gauss_min}' )

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
        ui.ERROR( '"fiber_shift" must be a scalar or a vector with 3 elements' )

    ui.INFO( f'Writing output connectome to "{output_connectome}"' )

    # load parcellation
    gm_nii = nib.load(parcellation_in)
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
    ui.INFO(f'Number of replicas for each streamline = {nReplicas}')

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
    ui.INFO(f'Threshold to use when computing assignments (in VOX space) = {thr:.3f}')

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
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # check if #(weights)==n_streamlines
        if weights_in is not None and n_streamlines!=w.size:
            ui.ERROR( f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) ' )

        zeros_count = 0

        with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
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
                        # if asgn[j][0] == 0 and asgn[j][1] == 0:
                        #     one_replica = np.array([[replicas_start[j][0]-0.5, replicas_start[j][1]-0.5, replicas_start[j][2]-0.5], 
                        #                             [replicas_end[j][0]-0.5, replicas_end[j][1]-0.5, replicas_end[j][2]-0.5]],
                        #                             dtype=np.float32)
                        #     TCK_out.write_streamline( one_replica, 2 )
                        #     str_count += 1
                        # if asgn[j][0] == 0 and asgn[j][1] > 0:
                        #     print(f'in connectome, replica = [{replicas_start[j][0]}, {replicas_start[j][1]}, {replicas_start[j][2]}')
                        #     one_replica = np.array([[replicas_start[j][0]-0.5, replicas_start[j][1]-0.5, replicas_start[j][2]-0.5]],
                        #                             dtype=np.float32)
                        #     TCK_out.write_streamline( one_replica, 1 )
                        #     str_count += 1
                        #     d_s = sqrt((replicas_start[j][0] - pts_start_vox[0,0])**2 + (replicas_start[j][1] - pts_start_vox[0,1])**2 + (replicas_start[j][2] - pts_start_vox[0,2])**2)
                        #     # print(f'dist_s: {d_s}, roi_pt = {gm_map[<int>(pts_start_vox[0,0]), <int>(pts_start_vox[0,1]), <int>(pts_start_vox[0,2])]}, roi_repl = {gm_map[<int>(replicas_start[j][0]), <int>(replicas_start[j][1]), <int>(replicas_start[j][2])]}, roi_asgn={asgn[j][0]}, roi_ret={roi_ret[0]}, pt = [{pts_start_vox[0,0]}, {pts_start_vox[0,1]}, {pts_start_vox[0,2]}], streamline_n = {i}')
                        # if asgn[j][1] == 0 and asgn[j][0] > 0:
                        #     one_replica = np.array([[replicas_end[j][0]-0.5, replicas_end[j][1]-0.5, replicas_end[j][2]-0.5]],
                        #                             dtype=np.float32)
                        #     TCK_out.write_streamline( one_replica, 1 )
                        #     str_count += 1
                        #     d_e = sqrt((replicas_end[j][0] - pts_end_vox[0,0])**2 + (replicas_end[j][1] - pts_end_vox[0,1])**2 + (replicas_end[j][2] - pts_end_vox[0,2])**2)
                        #     # print(f'dist_e: {d_e}, roi_pt = {gm_map[<int>(pts_end_vox[0,0]), <int>(pts_end_vox[0,1]), <int>(pts_end_vox[0,2])]}, roi_repl = {gm_map[<int>(replicas_end[j][0]), <int>(replicas_end[j][1]), <int>(replicas_end[j][2])]}, roi_asgn={asgn[j][1]}, roi_ret={roi_ret[1]},, pt = [{pts_start_vox[0,0]}, {pts_start_vox[0,1]}, {pts_start_vox[0,2]}], streamline_n = {i}')
                    zeros_count += (asgn.size - np.count_nonzero(asgn))

                    # find unique assignments and sum the weights of their replicas
                    asgn_sort = np.sort(asgn, axis=1) # shape = (nReplicas, 2)
                    asgn_unique = np.unique(asgn_sort, axis=0) 
                    weight_fraction = np.zeros(asgn_unique.shape[0], dtype=np.float64) # one value for each unique pair of ROI
                    for j in range(nReplicas):
                        idx = np.where(np.all(asgn_unique==asgn_sort[j],axis=1)) # find idx in weight_fraction corresponding to the pair of ROI of the current replica
                        weight_fraction[idx] += blurWeights_norm[j] # total fraction of the blurred streamline weight to be assigned to a specific pair of ROI

                    # update the connectome weights
                    weight_fraction = weight_fraction * w[i]
                    for k in range(asgn_unique.shape[0]):
                        if asgn_unique[k][0] == 0: continue
                        conn[asgn_unique[k][0]-1, asgn_unique[k][1]-1] += weight_fraction[k]

                pbar.update()

        if zeros_count > 0 : ui.WARNING(f'Some replicas are not assigned to any region (tot. {zeros_count})')


    except Exception as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        # if TCK_out is not None:
        #     TCK_out.close( write_eof=True, count=str_count )
        if symmetric:
            conn_sym = conn.T + conn
            np.fill_diagonal(conn_sym,np.diag(conn))
            if conn_out_ext=='.csv':
                np.savetxt(output_connectome, conn_sym, delimiter=",", fmt="%.5e")
            elif conn_out_ext=='.npy':
                np.save(output_connectome, conn_sym, allow_pickle=False)
            else:
                ui.ERROR( 'Invalid extension for the output connectome file' )
        else:
            if conn_out_ext=='.csv':
                np.savetxt(output_connectome, conn, delimiter=",", fmt="%.5e")
            elif conn_out_ext=='.npy':
                np.save(output_connectome, conn, allow_pickle=False)
            else:
                ui.ERROR( 'Invalid extension for the output connectome file' )
