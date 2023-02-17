#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import cython
import numpy as np
cimport numpy as np
import os
import nibabel as nib
from . import ui
# from tqdm import trange


def extract( input_dwi: str, input_scheme: str, output_dwi: str, output_scheme: str, b: list, b_step: float=0.0, verbose: bool=False, force: bool=False ):
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

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    ui.set_verbose( 2 if verbose else 1 )
    if not os.path.isfile(input_dwi):
        ui.ERROR( f'File "{input_dwi}" not found' )
    if not os.path.isfile(input_scheme):
        ui.ERROR( f'File "{input_scheme}" not found' )
    if not force:
        if os.path.isfile(output_dwi):
            ui.ERROR( 'Output DWI images already exist, use -f to overwrite' )
        if os.path.isfile(output_scheme):
            ui.ERROR( 'Output scheme already exists, use -f to overwrite' )

    try:
        # load the data
        niiDWI = nib.load( input_dwi )
        if niiDWI.ndim!=4:
            ui.ERROR( 'DWI data is not 4D' )

        # load the corresponding acquisition details
        scheme = np.loadtxt( 'DWI.txt' )
        if scheme.ndim!=2 or scheme.shape[1]!=4 or scheme.shape[0]!=niiDWI.shape[3]:
            ui.ERROR( 'DWI and scheme files are incorrect/incompatible' )
        bvals = scheme[:,3]

        # if requested, round the b-values
        if b_step>0.0:
            ui.INFO( f'Rounding b-values to nearest multiple of {b_step:.1f}' )
            bvals = np.round(bvals/b_step) * b_step

        # extract selected volumes
        idx = np.zeros_like( bvals, dtype=bool )
        for i in b:
            idx[ bvals==i ] = True
        n = np.count_nonzero(idx)
        ui.INFO( f'Extracted {n} volumes' )
        if n==0:
            ui.ERROR( 'The specified criterion selects 0 volumes' )
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
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )
