# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False

from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, File, Num, format_time
from dicelib.streamline import create_replicas
from dicelib.streamline cimport apply_affine
from dicelib.tractogram cimport LazyTractogram
from scipy.linalg import inv

import os

import nibabel as nib
import numpy as np

from time import time

logger = setup_logger('image')

def extract( input_dwi: str, input_scheme: str, output_dwi: str, output_scheme: str, b: list, b_step: float=0.0, verbose: int=3, force: bool=False ):
    """Extract volumes from a DWI dataset.

    Parameters
    ----------
    input_dwi : string
        Path to the file (.nii, .nii.gz) containing the data to process.

    input_scheme : string
        Input scheme file (text file).

    output_dwi : string
        Path to the file (.nii, .nii.gz) where to store the extracted volumes.

    b : list
        List of b-values to extract.

    b_step : float
        Round b-values to nearest integer multiple of b_step (default : don't round).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).

    force : boolean
        Force overwriting of the output (default : False).
    """
    set_verbose('image', verbose)

    files = [
        File(name='dwi_in', type_='input', path=input_dwi),
        File(name='scheme_in', type_='input', path=input_scheme),
        File(name='dwi_out', type_='output', path=output_dwi),
        File(name='scheme_out', type_='output', path=output_scheme)
    ]
    nums = [Num(name='round', value=b_step, min_=0.0)]
    if len(b) == 0:
        logger.error('No b-values specified')
    else:
        nums.extend([Num(name='b', value=i, min_=0.0) for i in b])
    check_params(files=files, nums=nums, force=force)

    logger.info('Extracting volumes from DWI dataset')
    t0 = time()

    try:
        # load the data
        niiDWI = nib.load( input_dwi )
        if niiDWI.ndim!=4:
            logger.error('DWI data is not 4D')

        # load the corresponding acquisition details
        scheme = np.loadtxt( 'DWI.txt' )
        if scheme.ndim!=2 or scheme.shape[1]!=4 or scheme.shape[0]!=niiDWI.shape[3]:
            logger.error('DWI and scheme files are incorrect/incompatible')
        bvals = scheme[:,3]

        # if requested, round the b-values
        if b_step>0.0:
            logger.subinfo(f'Rounding b-values to nearest multiple of {b_step:.1f}', indent_char='*')
            bvals = np.round(bvals/b_step) * b_step

        # extract selected volumes
        idx = np.zeros_like( bvals, dtype=bool )
        for i in b:
            idx[ bvals==i ] = True
        n = np.count_nonzero(idx)
        logger.subinfo(f'Number of extracted volumes: {n}', indent_char='*')
        if n==0:
            logger.error('The specified criterion selects 0 volumes')
        niiDWI_img = np.asanyarray(niiDWI.dataobj,dtype=niiDWI.get_data_dtype())[:,:,:,idx]
        scheme = scheme[idx,:]

        # save NIFTI file with only those volumes as well as the corresponding scheme file
        nib.Nifti1Image( niiDWI_img, niiDWI.affine ).to_filename( output_dwi )
        np.savetxt( output_scheme, scheme, fmt='%9.6f' )

    except Exception as e:
        if os.path.isfile( output_dwi ):
            os.remove( output_dwi )
        if os.path.isfile( output_scheme ):
            os.remove( output_scheme )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )




def tdi_ends(input_tractogram: str, input_ref: str, output_image: str, blur_core_extent: float=0.0, blur_gauss_extent: float=0.0, blur_spacing: float=0.25, blur_gauss_min: float=0.1, fiber_shift=0, verbose: int=3, force: bool=False):
    """Compute the TDI only for the ending points, possibly using blur.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    input_ref : string
        Path to the file containing the reference image.

    output_image : string
        Path to the file where to store the resulting image.

    blur_core_extent: float
        Extent of the core inside which the segments have equal contribution to the central one used by COMMITblur.

    blur_gauss_extent: float
        Extent of the gaussian damping at the border used by COMMITblur.

    blur_spacing : float
        To obtain the blur effect, streamlines are duplicated and organized in a cartesian grid;
        this parameter controls the spacing of the grid in mm (defaut : 0.25).

    blur_gauss_min: float
        Minimum value of the Gaussian to consider when computing the sigma (default : 0.1).

    fiber_shift : float or list of three float
        If necessary, apply a translation to streamline coordinates (default : 0) to account
        for differences between the reference system of the tracking algorithm and COMMIT.
        The value is specified in voxel units, eg 0.5 translates by half voxel.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('image', verbose)

    logger.info( 'Compute TDI of ending points' )
    t0 = time()

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='input_ref', type_='input', path=input_ref),
        File(name='output_image', type_='output', path=output_image)
    ]
    nums = [
        Num(name='blur_core_extent', value=blur_core_extent, min_=0.0),
        Num(name='blur_gauss_extent', value=blur_gauss_extent, min_=0.0),
        Num(name='blur_spacing', value=blur_spacing, min_=0.0, include_min=False),
        Num(name='blur_gauss_min', value=blur_gauss_min, min_=0.0, include_min=False)
    ]
    check_params(files=files, nums=nums, force=force)

    logger.subinfo( f'Input tractogram: "{input_tractogram}"', indent_char='*')
    logger.subinfo( f'Input reference: "{input_ref}"', indent_char='*')

    if blur_core_extent==0.0 and blur_gauss_extent==0.0:
        logger.subinfo('No blur will be applied', indent_char='*')
    else:
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

    # load reference image
    ref_nii = nib.load(input_ref)
    ref = ref_nii.get_fdata()
    ref_header = ref_nii.header
    affine = ref_nii.affine
    # cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm, dtype=np.int32)
    cdef float [:,::1] inverse = np.ascontiguousarray(inv(affine), dtype=np.float32) #inverse of affine
    cdef float [::1,:] M = inverse[:3, :3].T 
    cdef float [:] abc = inverse[:3, 3]
    cdef float [:] voxdims = np.asarray( ref_header.get_zooms(), dtype = np.float32 )

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

    # create image to fill
    tdi = np.zeros(ref.shape, dtype=np.float32)

    #----- iterate over input files -----
    TCK_in = None
    cdef size_t i, j, k = 0  
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo( f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*')

        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

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

                # increment the TDI image considering the replicas weights
                for j in range(nReplicas):
                    tdi[<int>replicas_start[j][0], <int>replicas_start[j][1], <int>replicas_start[j][2]] += blurWeights_norm[j]
                    tdi[<int>replicas_end[j][0], <int>replicas_end[j][1], <int>replicas_end[j][2]] += blurWeights_norm[j]
                pbar.update()

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        logger.subinfo( f'Output image: "{output_image}"', indent_char='*')
        nib.Nifti1Image( tdi, affine ).to_filename( output_image )
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )

