#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
import nibabel
from os.path import splitext, getsize


# Interface to actual C code
cdef extern from "processing_c.cpp":
    int do_spline_smoothing(
        float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len
    ) nogil


cpdef spline_smoothing( filename_tractogram, filename_tractogram_out=None, control_point_ratio=0.25, segment_len=1.0, verbose=False ) :
    """Smooth each streamline in the input tractogram using Catmull-Rom splines.
       More info at http://algorithmist.net/docs/catmullrom.pdf.

    Parameters
    ----------
    filename_tractogram : string
        Path to the file (.trk or .tck) containing the streamlines to process.

    filename_tractogram_out : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_smooth' to the input filename.

    control_point_ratio : float
        Percent of control points to use in the interpolating spline (default : 0.25).

    segment_len : float
        Sampling resolution of the final streamline after interpolation (default : 1.0).

    verbose : boolean
        Print information and progess (default : False).
    """

    try :
        basename, extension = splitext(filename_tractogram)
        tractogram_in = nibabel.streamlines.load( filename_tractogram, lazy_load=True )
        hdr = tractogram_in.header
        if extension == ".trk":
            n_count = int( hdr['nb_streamlines'] )
        else:
            n_count = int( hdr['count'] )
    except :
        raise IOError( 'Track file not found' )

    if control_point_ratio < 0 or control_point_ratio > 1 :
        raise ValueError( "'control_point_ratio' parameter must be in [0..1]" )
    
    if filename_tractogram_out is None :
        filename_tractogram_out = basename+'_smooth'+extension

    if verbose :
        print( '* input tractogram :' )
        print( f'\t- {filename_tractogram}' )
        print( f'\t- {n_count} streamlines' )
        
        mb = getsize( filename_tractogram )/1.0E6
        if mb >= 1E3:
            print( f'\t- {mb/1.0E3:.2f} GB' )
        else:            
            print( f'\t- {mb:.2f} MB' )
        
        print( '* output tractogram :' )
        print( f'\t- {filename_tractogram_out}' )
        print( f'\t- control points : {control_point_ratio*100.0:.1f}%')
        print( f'\t- segment length : {segment_len:.2f}' )

    # create the structure for the input and output polyline
    cdef float [:, ::1] npaFiberI
    cdef float* ptr_npaFiberI
    cdef float [:, ::1] npaFiberO = np.ascontiguousarray( np.zeros( (3*10000,1) ).astype(np.float32) )
    cdef float* ptr_npaFiberO = &npaFiberO[0,0]

    streamlines_out = []
    for f in tractogram_in.streamlines:
        npaFiberI = np.ascontiguousarray( f.copy() )
        ptr_npaFiberI = &npaFiberI[0,0]

        n = do_spline_smoothing( ptr_npaFiberI, f.shape[0], ptr_npaFiberO, control_point_ratio, segment_len )

        streamlines_out.append( np.reshape( npaFiberO[:3*n].copy(), (n,3) ) )

    tractogram_out = nibabel.streamlines.Tractogram(streamlines_out, affine_to_rasmm=tractogram_in.tractogram.affine_to_rasmm)
    nibabel.streamlines.save(tractogram_out, filename_tractogram_out)

    if verbose :
        mb = getsize( filename_tractogram_out )/1.0E6
        if mb >= 1E3:
            print( f'\t- {mb/1.0E3:.2f} GB' )
        else:            
            print( f'\t- {mb:.2f} MB' )
