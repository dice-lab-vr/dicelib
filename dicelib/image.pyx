# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False

from dicelib.ui import set_verbose, setup_logger
from dicelib.utils import check_params, File, Num, format_time

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
    check_params(files, nums, force)

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
        logger.subinfo(f'Extracted {n} volumes', indent_char='*')
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
