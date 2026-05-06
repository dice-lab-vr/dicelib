# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
from concurrent.futures import ThreadPoolExecutor
from libc.math cimport round as cround, ceil as cceil, sqrt, INFINITY
from libcpp cimport bool
import nibabel as nib
import numpy as np
import os
from scipy.linalg import inv
from time import time
from dicelib.streamline import create_replicas
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, File, Num, format_time
from dicelib.streamline cimport apply_xform_to_point
from dicelib.tractogram cimport LazyTractogram

logger = setup_logger('connectivity')


def compute_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


cdef compute_grid( float thr ):
    """Compute the offsets grid
        Parameters
        ---------------------
        thr : float
            Radius of the radial search (in voxel units)
    """
    cdef:
        int thr_grid = <int>cceil(thr)
        int[:] dist_grid

    # create the mesh
    mesh = np.linspace( -thr_grid, thr_grid, 2*thr_grid+1 )
    mx, my, mz = np.meshgrid( mesh, mesh, mesh )

    # find the centers of each voxels
    centers = np.stack([mx.ravel(), my.ravel(), mz.ravel()], axis=1)

    # sort the centers based on their distance from grid_center
    dist_grid = (centers**2).sum(axis=1).argsort().astype(np.int32)
    return centers[ dist_grid ].astype(np.float32)


cdef int radial_search( float [:] p, int[:,:,::1] label_img, float thr=0, float [:,::1] grid=None, int[:] count_neighbours=None ) noexcept nogil:
    """Compute the label corresponding to a point.

    Parameters
    ----------
    p : 3x1 float array
        3D point to evaluate.
    label_img : 3D numpy array
        3D voxelwise image containing the labels.
    thr : float
        Maximum radius [in mm] of the search.
    grid : Nx3 numpy array
        Precomputed grid of the voxels to check.

    Returns
    -------
    out_label : int
        Label assigned to the point.
    """
    cdef:
        float px=p[0], py=p[1], pz=p[2]
        float x, y, z
        int vx, vy, vz
        float dist, dist_tmp=INFINITY
        int layer=0
        int out_label=0
        size_t i

    if thr < 0.5 or grid is None:
        # no radial serach, check only the underlying voxel
        vx = <int>cround(px)
        vy = <int>cround(py)
        vz = <int>cround(pz)
        if vx < 0 or vx >= label_img.shape[0] or vy < 0 or vy >= label_img.shape[1] or vz < 0 or vz >= label_img.shape[2]:
            return 0
        return label_img[vx, vy, vz]
    else:
        # iterate over all grid voxels
        for i in xrange(grid.shape[0]):
            # check if the voxel is inside the mask
            vx = <int>cround(px + grid[i][0])
            vy = <int>cround(py + grid[i][1])
            vz = <int>cround(pz + grid[i][2])
            if vx < 0 or vx >= label_img.shape[0] or vy < 0 or vy >= label_img.shape[1] or vz < 0 or vz >= label_img.shape[2]:
                continue
            if label_img[vx, vy, vz]<=0:
                continue

            # compute distance
            x = max(vx-0.5-px, 0, px-vx-0.5)
            y = max(vy-0.5-py, 0, py-vy-0.5)
            z = max(vz-0.5-pz, 0, pz-vz-0.5)
            dist = sqrt(x*x + y*y + z*z)
            if dist <= thr and dist < dist_tmp:
                out_label = label_img[vx, vy, vz]
                dist_tmp = dist

            if i == count_neighbours[layer]:
                if dist_tmp<INFINITY:
                    break
                else:
                    layer += 1
    return out_label


cpdef assign(tractogram_filename: str, atlas_filename: str, out_assignments_filename: str, distance: float=2.0, n_threads: int=None, force: bool=False, verbose: int=3, log_list=None) :
    """Compute the assignments of the streamlines based on an atlas (i.e. label file).

    A radial search from each streamline endpoint is performed to locate the nearest node
    and assign the streamline to the corresponding pair of labels; the parameter `distance`
    controls the maximum radius in mm of the search.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    atlas_filename : str
        Path to the file (.nii, .nii.gz) containing the labels of the atlas.
    out_assignments_filename : str
        Path to the file (.txt, .npy) where to store the resulting assignments.
    distance : float, default=2.0
        Distance [in mm] to consider in the radial search when computing the assignments.
    n_threads : int, deault=None
        How many threads to use in parallel for the computations;
        if not specfied, all available threads will be used.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('connectivity', verbose)
    logger.info(f'Computing assignments')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck']),
        File(name='atlas_filename', type_='input', path=atlas_filename, ext=['.nii', '.nii.gz']),
        File(name='out_assignments_filename', type_='output', path=out_assignments_filename, ext=['.txt', '.npy'])
    ]
    nums = [
        Num(name='distance', value=distance, min_=0.0, include_min=True)
    ]
    if n_threads is not None:
        nums.append(Num(name='n_threads', value=n_threads, min_=1))
    check_params(files=files, nums=nums, force=force)

    num_streamlines = int(LazyTractogram(tractogram_filename, mode='r').header["count"])
    logger.subinfo(f'Number of input streamlines: {num_streamlines}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'Distance threshold: {distance} mm', indent_char='*', indent_lvl=1)

    # Load of the gm map
    gm_map_img = nib.load(atlas_filename)
    gm_map_data = gm_map_img.get_fdata()
    gm_map_dtype = gm_map_img.header.get_data_dtype()
    if gm_map_dtype.char not in ['b',' h', 'i', 'l', 'B', 'H', 'I', 'L']:
        warning_msg = f'Atlas data type is \'{gm_map_dtype}\'. It is recommended to use an integer data type.'
        logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)

    if num_streamlines > 3:
        if n_threads:
            MAX_THREAD = n_threads
        else:
            MAX_THREAD = os.cpu_count()
    else:
        MAX_THREAD = 1
    chunk_size = int(num_streamlines / MAX_THREAD)
    chunk_groups = [e for e in compute_chunks(np.arange(num_streamlines), chunk_size)]
    chunks_asgn = []

    pbar_array = np.zeros(MAX_THREAD, dtype=np.int32)
    with ProgressBar(multithread_progress=pbar_array, total=num_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_THREAD) as executor:
            future = [
                executor.submit(
                    _assign,
                    tractogram_filename,
                    pbar_array,
                    i,
                    start_chunk=int(chunk_groups[i][0]),
                    end_chunk=int(chunk_groups[i][len(chunk_groups[i]) - 1] + 1),
                    gm_map_data=gm_map_data,
                    gm_map_img=gm_map_img,
                    threshold=distance) for i in range(len(chunk_groups))
            ]
            chunks_asgn = [f.result() for f in future]
            chunks_asgn = [c for f in chunks_asgn for c in f]

    out_assignments_ext = os.path.splitext(out_assignments_filename)[1]
    if out_assignments_ext == '.txt':
        with open(out_assignments_filename, "w") as text_file:
            for reg in chunks_asgn:
                print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
    else:
        np.save(out_assignments_filename, chunks_asgn, allow_pickle=False)

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef _assign( input_tractogram: str, int[:] pbar_array, int id_chunk, int start_chunk, int end_chunk, gm_map_data, gm_map_img, threshold: 2.0 ):
    cdef:
        int [:,:,::1] gm_map = np.ascontiguousarray(gm_map_data, dtype=np.int32)
        double [:,::1] affine_inv = np.linalg.inv(gm_map_img.affine)
        float thr = <float> threshold/np.max(gm_map_img.header.get_zooms())
        int n_streamlines = end_chunk - start_chunk
        int[:,:] assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
        float [:] p = np.zeros(3, dtype=np.float32)
        float [:,::1] grid
        cdef int[:] count_neighbours
        size_t i

    # compute the grid of voxels to check
    grid = compute_grid(thr)
    layers = np.arange(0, <int>cceil(thr)+1, 1) # e.g [0, 1, 2, 3]
    lato = layers * 2 + 1 # e.g [0, 3, 5, 7] = layerx2+1
    neighbs = [v**3-1 for v in lato] # e.g [1, 27, 125, 343] = (lato)**3
    count_neighbours = np.array(neighbs, dtype=np.int32)

    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    with nogil:
        while i < start_chunk:
            TCK_in.read_streamline()
            i += 1
        for i in xrange( n_streamlines ):
            TCK_in.read_streamline()
            apply_xform_to_point( TCK_in.streamline[0,:], affine_inv, p )
            assignments[i,0] = radial_search( p, gm_map, thr, grid, count_neighbours )
            apply_xform_to_point( TCK_in.streamline[TCK_in.n_pts-1,:], affine_inv, p )
            assignments[i,1] = radial_search( p, gm_map, thr, grid, count_neighbours )
            pbar_array[id_chunk] += 1
    if TCK_in is not None:
        TCK_in.close()
    return assignments


def compute_connectome_blur(input_tractogram: str, output_connectome: str, weights_in: str, input_nodes: str,
                            blur_core_extent: float, blur_gauss_extent: float, blur_spacing: float=0.25,
                            blur_gauss_min: float=0.1, offset_thr: float=0.0, symmetric: bool=False, fiber_shift=0,
                            verbose: int=3, force: bool=False):
    """Build the connectome weighted by COMMITblur (only sum).

    Parameters
    ----------
    input_tractogram : str
        Path to the file (.tck) containing the streamlines to process.

    output_connectome : str
        Path to the file where to store the resulting connectome.

    weights_in : str
        Scalar file (.txt, .npy) for the input streamline weights estimated by COMMITblur.

    input_nodes : str
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

    set_verbose('connectivity', verbose)

    logger.info( 'Compute connectome weighted by COMMITblur' )
    t0 = time()

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
    logger.subinfo( f'blur_core_extent:  {blur_core_extent}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'blur_gauss_extent: {blur_gauss_extent}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'blur_spacing:      {blur_spacing}', indent_lvl=1, indent_char='-')
    logger.subinfo( f'blur_gauss_min:    {blur_gauss_min}', indent_lvl=1, indent_char='-')

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
    if nReplicas > 0:
        logger.subinfo(f'Number of replicas for each streamline: {nReplicas}', indent_lvl=1, indent_char='-')

    # compute the grid of voxels for the radial search
    threshold = core_extent + gauss_extent
    # print(f'thr = {thr}')
    cdef float thr = threshold + (offset_thr/np.max(voxdims)) # if input streamlines are all connecting but using a radial search
    grid = compute_grid( thr )
    layers = np.arange( 0,<int> cceil(thr)+1, 1 ) # e.g. layer=[0, 1, 2, 3]
    lato = layers * 2 + 1 # e.g. lato = [0, 3, 5, 7] = layerx2+1
    neighbs = [v**3-1 for v in lato] # e.g. [1, 27, 125, 343] = (lato)**3
    cdef int[:] count_neighbours = np.array(neighbs, dtype=np.int32)
    thr += 0.005 # to take into accound rounding errors in the distance of the replicas
    # print(f'core+gauss = {core_extent + gauss_extent}')
    logger.subinfo(f'Threshold to use when computing assignments (in VOX space): {thr:.3f}', indent_lvl=1, indent_char='-')

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
    cdef int [:] start_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] end_vox   = np.zeros(3, dtype=np.int32)
    cdef int [:] roi_ret   = np.array([0,0], dtype=np.int32)
    cdef float [:,::1] points_mat = np.zeros( (2,3), dtype=np.float32)

    # create connectome to fill
    n_rois = np.max(gm).astype(np.int32)
    conn = np.zeros((n_rois, n_rois), dtype=np.float64)

    #----- iterate over input files -----
    TCK_in = None
    cdef size_t i, j, k = 0
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo( f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*')

        # check if #(weights)==n_streamlines
        if n_streamlines!=w.size:
            logger.error(f'Number of weights ({w.size}) is different from the number of streamline ({n_streamlines})')

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
                    #FIXME: replace 'apply_affine' with 'apply_xform_to_point'
                    # pts_start_vox = apply_affine(pts_start, M, abc, pts_start_tmp) # starting points in voxel space
                    # pts_end_vox   = apply_affine(pts_end,   M, abc, pts_end_tmp)   # ending points in voxel space

                    # create replicas of starting and ending points
                    replicas_start = create_replicas(pts_start_vox, blurRho, blurAngle, nReplicas, fiber_shiftX, fiber_shiftY, fiber_shiftZ)
                    replicas_end   = create_replicas(pts_end_vox,   blurRho, blurAngle, nReplicas, fiber_shiftX, fiber_shiftY, fiber_shiftZ)

                    # compute assignments of the replicas
                    for j in range(nReplicas):
                        points_mat = np.array([[replicas_start[j][0], replicas_start[j][1], replicas_start[j][2]],
                                                [replicas_end[j][0], replicas_end[j][1], replicas_end[j][2]]],
                                                dtype=np.float32)
                        #FIXME: sistemare la call alla funzione
                        # asgn_view[j][:] = streamline_assignment( start_vox, end_vox, roi_ret, points_mat, grid, gm_map, thr, count_neighbours)

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
    logger.subinfo( f'Output connectome: "{output_connectome}"', indent_char='*')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )



def build_connectome( input_assignments: str, output_connectome: str, input_weights: str=None, input_tractogram: str=None, input_nodes: str=None, atlas_dist: float=2.0, metric: str='sum', symmetric: bool=False, n_threads: int=None, verbose: int=3, force: bool=False, log_list=None ):
    """Build the (weighted) connectome having the assignments or the tractogram and an atlas.

    Parameters
    ----------
    input_weights : str
        Scalar file (.txt, .npy) for the input streamline weights.

    input_assignments : str
        Path to the file (.txt, .npy) containing the streamline assignments.

    output_connectome : str
        Path to the file where to store the resulting connectome.

    input_tractogram : str
        Path to the file (.tck) containing the streamlines to process.

    input_nodes : str
        Path to the file containing the gray matter parcellation (nodes of the connectome).

    atlas_dist : float
        Distance [in mm] used to assign streamlines to the atlas' nodes (default: 2.0).

    metric : str
        Operation to compute the value of the edges, options: sum, mean, min, max (default: sum).

    symmetric : boolean
        Make output connectome symmetric (default : False).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('connectivity', verbose)
    logger.info('Computing connectome')
    t0 = time()

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
            logger.error(f'Tractogram file not provided. Required if the assignments does not exist.')
        if input_nodes is None:
            logger.error(f'Nodes file not provided. Required if the assignments does not exist.')
        files.extend([
            File(name='tractogram_in', type_='input', path=input_tractogram, ext=['.tck']),
            File(name='nodes_in', type_='input', path=input_nodes, ext=['.nii', '.nii.gz'])
        ])

        # logger.info('No assignments file found. Computing assignments')
        logger.subinfo(f'Input tractogram: \'{input_tractogram}\'', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Input parcellation: \'{input_nodes}\'', indent_char='*', indent_lvl=1)

        # compute assignments
        log_list2 = []
        ret_subinfo2 = logger.subinfo('Computing assignments', indent_lvl=1, indent_char='*', with_progress=verbose>2)
        with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo2, log_list=log_list2) as pbar:
            assign(input_tractogram, input_nodes, input_assignments, atlas_dist, verbose=1, n_threads=n_threads, log_list=log_list2)
        set_verbose('connectivity', verbose)

    check_params(files=files, force=force)

    # streamline assignments
    input_assignments_ext = os.path.splitext(input_assignments)[1]
    if input_assignments_ext=='.txt':
        asgn = np.loadtxt( input_assignments ).astype(np.int32)
    else:
        asgn = np.load( input_assignments, allow_pickle=False ).astype(np.int32)
    n_streamlines = asgn.shape[0]
    asgn_sort = np.sort(asgn, axis=1) # shape = (n_streamlines, 2)

    # check if the assignments match with the number of streamlines in the tractogram
    if input_tractogram is not None:
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_str_tck = int( TCK_in.header['count'] )
        TCK_in.close()
        if n_streamlines != n_str_tck:
            logger.error(f'Number of streamlines in the tractogram ({n_str_tck}) is different from the number of streamline assignments ({n_streamlines})')

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
            logger.error(f'Number of weights ({w.size}) is different from the number of streamline assignments ({n_streamlines})')

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
        warning_msg = f'Number of non-connecting streamlines {count_unconn}'
        logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)

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

    logger.subinfo( f'Output connectome: "{output_connectome}"', indent_char='*', indent_lvl=1)
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )