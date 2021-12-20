#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
import sys, os, glob, random
from lazytck import LazyTCK
import ui
from tqdm import trange
from libc.math cimport sqrt


# Interface to actual C code
cdef extern from "tractogram.hpp":
    int do_spline_smoothing(
        float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len
    ) nogil


cpdef streamline_length( float [:,:] streamline, int n=0 ):
    """Compute the length of a streamline.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data
    n : int
        Writes first n points of the streamline. If n<=0 (default), writes all points

    Returns
    -------
    length : double
        Length of the streamline in mm
    """
    if n<0:
        n = streamline.shape[0]
    cdef float* ptr     = &streamline[0,0]
    cdef float* ptr_end = ptr+n*3-3
    cdef double length = 0.0
    while ptr<ptr_end:
        length += sqrt( (ptr[3]-ptr[0])**2 + (ptr[4]-ptr[1])**2 + (ptr[5]-ptr[2])**2 )
        ptr += 3
    return length


def compute_lenghts( input_tractogram: str, output_scalar_file: str, verbose: bool=False, force: bool=False ):
    """Compute the lenghts os the streamlines in a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_scalar_file : string
        Path to the file where to store the computed streamline lenghts.

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    ui.set_verbose( 2 if verbose else 1 )
    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if os.path.isfile(output_scalar_file) and not force:
        ui.ERROR( 'Output scalar file already exists, use -f to overwrite' )

    #----- iterate over input streamlines -----
    TCK_in = None
    try:
        # open the input file
        TCK_in = LazyTCK( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        if verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )

        lengths = np.empty( n_streamlines, dtype=np.double )
        if n_streamlines>0:
            for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
        np.savetxt( output_scalar_file, lengths, fmt='%.4f' )

        if verbose and n_streamlines>0:
            ui.INFO( f'min={lengths.min():.3f}   max={lengths.max():.3f}   mean={lengths.mean():.3f}   std={lengths.std():.3f}' )


    except BaseException as e:
        if os.path.isfile( output_scalar_file ):
            os.remove( output_scalar_file )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()


def info( input_tractogram: str, compute_lengts: bool=False ):
    """Print some information about a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    compute_lengts : boolean
        Show stats on streamline lenghts (default : False).
    """
    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )

    #----- iterate over input streamlines -----
    TCK_in  = None
    try:
        # open the input file
        TCK_in = LazyTCK( input_tractogram, mode='r' )

        # print the header
        ui.INFO( 'HEADER content')
        max_len = max([len(k) for k in TCK_in.header.keys()])
        for k, v in TCK_in.header.items():
            print( ui.hWhite+ '%0*s'%(max_len,k) +ui.Reset+ui.fWhite+ ':  ' + v +ui.Reset )
        print( '' )

        # print stats on lengths
        if compute_lengts:
            ui.INFO( 'Streamline lenghts')
            n_streamlines = int( TCK_in.header['count'] )
            if n_streamlines>0:
                lengths = np.empty( n_streamlines, dtype=np.double )
                for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading
                    lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                print( f'   {ui.hWhite}min{ui.Reset}{ui.fWhite}={lengths.min():.3f}   {ui.hWhite}max{ui.Reset}{ui.fWhite}={lengths.max():.3f}   {ui.hWhite}mean{ui.Reset}{ui.fWhite}={lengths.mean():.3f}   {ui.hWhite}std{ui.Reset}{ui.fWhite}={lengths.std():.3f}{ui.Reset}' )
            else:
                ui.WARNING( 'The tractogram is empty' )

    except BaseException as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()


def filter( input_tractogram: str, output_tractogram: str, minlength: float=None, maxlength: float=None, minweight: float=None, maxweight: float=None, weights_in: str=None, weights_out: str=None, verbose: bool=False, force: bool=False ):
    """Filter out the streamlines in a tractogram according to some criteria.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram.

    minlength : float
        Keep streamlines with length [in mm] >= this value.

    maxlength : float
        Keep streamlines with length [in mm] <= this value.

    minweight : float
       Keep streamlines with weight >= this value.

    maxweight : float
        Keep streamlines with weight <= this value.

    weights_in : str
        Text file with the input streamline w.

    weights_out : str
        Text file with the output streamline w.

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    ui.set_verbose( 2 if verbose else 1 )

    n_wrote = 0
    TCK_in  = None
    TCK_out = None

    # check input
    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if os.path.isfile(output_tractogram) and not force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )

    if minlength is not None:
        if minlength<0:
            ui.ERROR( '"minlength" must be >= 0' )
        ui.INFO( f'Keep streamlines with length >= {minlength} mm' )
    if maxlength is not None:
        if maxlength<0:
            ui.ERROR( '"maxlength" must be >= 0' )
        if minlength and minlength>maxlength:
            ui.ERROR( '"minlength" must be <= "maxlength"' )
        ui.INFO( f'Keep streamlines with length <= {maxlength} mm' )

    # read the streamline weights (if any)
    if weights_in is not None:
        if not os.path.isfile( weights_in ):
            ui.ERROR( f'File "{weights_in}" not found' )
        w = np.loadtxt( weights_in )
        ui.INFO( 'Using streamline weights from text file' )
        if minweight is not None and minweight<0:
            ui.ERROR( '"minweight" must be >= 0' )
        ui.INFO( f'Keep streamlines with weight >= {minweight} mm' )
        if maxweight is not None and maxweight<0:
            ui.ERROR( '"maxweight" must be >= 0' )
        if minweight is not None and minweight>maxweight:
            ui.ERROR( '"minweight" must be <= "maxweight"' )
        ui.INFO( f'Keep streamlines with weight <= {maxweight} mm' )
    else:
        w = np.array( [] )

    #----- iterate over input streamlines -----
    try:
        # open the input file
        TCK_in = LazyTCK( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # check if #(weights)==n_streamlines
        if weights_in is not None and n_streamlines!=w.size:
            ui.ERROR( f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) ' )

        # open the outut file
        TCK_out = LazyTCK( output_tractogram, mode='w', header=TCK_in.header )

        kept = np.ones( n_streamlines, dtype=bool )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading

            # filter by length
            if minlength is not None or maxlength is not None:
                length = streamline_length(TCK_in.streamline, TCK_in.n_pts)
                if minlength is not None and length<minlength :
                    kept[i] = False
                    continue
                if maxlength is not None and length>maxlength :
                    kept[i] = False
                    continue

            # filter by weight
            if weights_in is not None and (
                (minweight is not None and w[i]<minweight) or
                (maxweight is not None and w[i]>maxweight)
            ):
                kept[i] = False
                continue

            # write streamline to output file
            TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )

        if weights_out is not None and w.size>0:
            np.savetxt( weights_out, w[kept==True], fmt='%.5e' )

        n_wrote = np.count_nonzero( kept )
        ui.INFO( f'{n_wrote} streamlines in output tractogram' )

    except BaseException as e:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( output_tractogram ):
            os.remove( output_tractogram )
        if weights_out and os.path.isfile( weights_out ):
            os.remove( weights_out )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( n_wrote )


def split( input_tractogram: str, filename_assignments: str, output_folder: str='bundles', max_open: int=None, verbose: bool=False, force: bool=False ):
    """Split the streamlines in a tractogram according to an assignment file.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to split.

    filename_assignments : string
        File containing the streamline assignments (two numbers/row); these can be stored as
        either a simple .txt file or according to the NUMPY format (.npy), which is faster.

    output_folder : string
        Output folder for the splitted tractograms.

    max_open : integer
        Maximum number of files opened at the same time (default : 50% of SC_OPEN_MAX system variable).

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    ui.set_verbose( 2 if verbose else 1 )

    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if not os.path.isfile(filename_assignments):
        ui.ERROR( f'File "{filename_assignments}" not found' )
    if not os.path.isdir(output_folder):
        os.mkdir( output_folder )
    else:
        if force:
            for f in glob.glob( os.path.join(output_folder,'*.tck') ):
                os.remove(f)
        else:
            ui.ERROR( 'Output folder already exists, use -f to overwrite' )
    ui.INFO( f'Writing output tractograms to "{output_folder}"' )

    if max_open is None:
        max_open = int( os.sysconf('SC_OPEN_MAX')*0.5 )
    ui.INFO( f'Using {max_open} files open simultaneously' )

    #----- iterate over input streamlines -----
    TCK_in        = None
    TCK_out       = None
    TCK_outs      = {}
    TCK_outs_size = {}
    n_wrote       = 0
    try:
        # open the tractogram
        TCK_in = LazyTCK( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # open the assignments
        if os.path.splitext(filename_assignments)[1]=='.txt':
            assignments = np.loadtxt( filename_assignments, dtype=np.int32 )
        elif os.path.splitext(filename_assignments)[1]=='.npy':
            assignments = np.load( filename_assignments, allow_pickle=False ).astype(np.int32)
        else:
            ui.ERROR( 'Not a valid extension for the assignments file' )
        if assignments.ndim!=2 or assignments.shape[1]!=2:
            ui.ERROR( 'Unable to open assignments file' )
        ui.INFO( f'{assignments.shape[0]} assignments in input file' )

        # check if #(assignments)==n_streamlines
        if n_streamlines!=assignments.shape[0]:
            ui.ERROR( f'# of assignments ({assignments.shape[0]}) is different from # of streamlines ({n_streamlines}) ' )

        # create empty tractograms for unique assignments
        unique_assignments = np.unique(assignments, axis=0)
        for i in range( unique_assignments.shape[0] ):
            if unique_assignments[i,0]==0 or unique_assignments[i,1]==0:
                continue
            if unique_assignments[i,0] <= unique_assignments[i,1]:
                key = f'{unique_assignments[i,0]}-{unique_assignments[i,1]}'
            else:
                key = f'{unique_assignments[i,1]}-{unique_assignments[i,0]}'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTCK( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )

        # add key for non-connecting streamlines
        key = 'unassigned'
        TCK_outs[key] = None
        TCK_outs_size[key] = 0
        tmp = LazyTCK( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
        tmp.close( write_eof=False, count=0 )

        ui.INFO( f'Created {len(TCK_outs)} empty files for output tractograms' )

        #----  iterate over input streamlines  -----
        n_file_open = 0
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            # get the key of the dictionary
            if assignments[i,0]==0 or assignments[i,1]==0:
                key = 'unassigned'
            elif assignments[i,0] <= assignments[i,1]:
                key = f'{assignments[i,0]}-{assignments[i,1]}'
            else:
                key = f'{assignments[i,1]}-{assignments[i,0]}'

            # check if need to open file
            if TCK_outs[key] is None:
                fname = os.path.join(output_folder,f'{key}.tck')
                if n_file_open==max_open:
                    key_to_close = random.choice( [k for k,v in TCK_outs.items() if v!=None] )
                    TCK_outs[key_to_close].close( write_eof=False )
                    TCK_outs[key_to_close] = None
                else:
                    n_file_open += 1

                TCK_outs[key] = LazyTCK( fname, mode='a' )

            # write input streamline to correct output file
            TCK_outs[key].write_streamline( TCK_in.streamline, TCK_in.n_pts )
            TCK_outs_size[key] += 1
            n_wrote += 1

        ui.INFO( f'{n_wrote-TCK_outs_size["unassigned"]} connecting, {TCK_outs_size["unassigned"]} non-connecting' )

    except BaseException as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        ui.INFO( 'Closing files and update totals' )
        if TCK_in is not None:
            TCK_in.close()
        for key in TCK_outs.keys():
            if TCK_outs[key] is not None:
                TCK_outs[key].close( write_eof=False )
            # Update 'count' and write EOF marker
            tmp = LazyTCK( os.path.join(output_folder,f'{key}.tck'), mode='a' )
            tmp.close( write_eof=True, count=TCK_outs_size[key] )


cpdef spline_smoothing( input_tractogram, output_tractogram=None, control_point_ratio=0.25, segment_len=1.0, verbose=False, force=False ):
    """Smooth each streamline in the input tractogram using Catmull-Rom splines.
       More info at http://algorithmist.net/docs/catmullrom.pdf.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_smooth' to the input filename.

    control_point_ratio : float
        Percent of control points to use in the interpolating spline (default : 0.25).

    segment_len : float
        Sampling resolution of the final streamline after interpolation (default : 1.0).

    verbose : boolean
        Print information messages (default : False).

    force : boolean
        Force overwriting of the output (default : False).
    """
    cdef float [:,:] npaFiberI
    cdef float [:,:] npaFiberO

    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if os.path.isfile(output_tractogram) and not force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )

    if control_point_ratio <= 0 or control_point_ratio > 1 :
        raise ValueError( "'control_point_ratio' parameter must be in (0..1]" )

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_smooth'+extension

    try:
        TCK_in = LazyTCK( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        TCK_out = LazyTCK( output_tractogram, mode='w', header=TCK_in.header )

        if verbose :
            ui.INFO( 'Input tractogram :' )
            ui.INFO( f'\t- {input_tractogram}' )
            ui.INFO( f'\t- {n_streamlines} streamlines' )

            mb = os.path.getsize( input_tractogram )/1.0E6
            if mb >= 1E3:
                ui.INFO( f'\t- {mb/1.0E3:.2f} GB' )
            else:
                ui.INFO( f'\t- {mb:.2f} MB' )

            ui.INFO( 'Output tractogram :' )
            ui.INFO( f'\t- {output_tractogram}' )
            ui.INFO( f'\t- control points : {control_point_ratio*100.0:.1f}%')
            ui.INFO( f'\t- segment length : {segment_len:.2f}' )

        # process each streamline
        npaFiberI = TCK_in.streamline
        npaFiberO = np.empty( (3000,3), dtype=np.float32 )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            n = do_spline_smoothing( &npaFiberI[0,0], TCK_in.n_pts, &npaFiberO[0,0], control_point_ratio, segment_len )
            TCK_out.write_streamline( npaFiberO, n )

    except BaseException as e:
        TCK_out.close()
        if os.path.exists( output_tractogram ):
            os.remove( output_tractogram )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        TCK_in.close()
        TCK_out.close()

    if verbose :
        mb = os.path.getsize( output_tractogram )/1.0E6
        if mb >= 1E3:
            ui.INFO( f'\t- {mb/1.0E3:.2f} GB' )
        else:
            ui.INFO( f'\t- {mb:.2f} MB' )