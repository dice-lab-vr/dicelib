#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
import sys, os
from os.path import splitext, getsize
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui
from tqdm import trange

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
    cdef float [:,:] npaFiberI
    cdef float [:,:] npaFiberO

    if control_point_ratio <= 0 or control_point_ratio > 1 :
        ui.ERROR( "'control_point_ratio' parameter must be in (0..1]" )
    
    if filename_tractogram_out is None :
        basename, extension = splitext(filename_tractogram)
        filename_tractogram_out = basename+'_smooth'+extension

    try:
        TCK_in  = LazyTCK( filename_tractogram, read_mode=True )
        if 'count' in TCK_in.header.keys():
            n_streamlines = int( TCK_in.header['count'] )
        else:
            # TODO: allow the possibility to wotk also in this case
            ui.ERROR( '"count" field not found in header' )
            sys.exit(1)

        TCK_out = LazyTCK( filename_tractogram_out, read_mode=False, header=TCK_in.header )

        if verbose :
            ui.LOG( 'Input tractogram :' )
            ui.LOG( f'\t- {filename_tractogram}' )
            ui.LOG( f'\t- {n_streamlines} streamlines' )
            
            mb = getsize( filename_tractogram )/1.0E6
            if mb >= 1E3:
                ui.LOG( f'\t- {mb/1.0E3:.2f} GB' )
            else:            
                ui.LOG( f'\t- {mb:.2f} MB' )
            
            ui.LOG( 'Output tractogram :' )
            ui.LOG( f'\t- {filename_tractogram_out}' )
            ui.LOG( f'\t- control points : {control_point_ratio*100.0:.1f}%')
            ui.LOG( f'\t- segment length : {segment_len:.2f}' )

        # process each streamline
        npaFiberI = TCK_in.streamline
        npaFiberO = np.empty( (3000,3), dtype=np.float32 )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            n = do_spline_smoothing( &npaFiberI[0,0], TCK_in.n_pts, &npaFiberO[0,0], control_point_ratio, segment_len )
            TCK_out.write_streamline( npaFiberO, n )

    except:
        TCK_out.close()
        if os.path.exists( filename_tractogram_out ):
            os.remove( filename_tractogram_out )
        ui.ERROR( 'Unable to smooth streamlines in the tractogram' )

    finally:
        TCK_in.close()
        TCK_out.close()


    if verbose :
        mb = getsize( filename_tractogram_out )/1.0E6
        if mb >= 1E3:
            ui.LOG( f'\t- {mb/1.0E3:.2f} GB' )
        else:            
            ui.LOG( f'\t- {mb:.2f} MB' )
