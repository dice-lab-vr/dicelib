#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import cython
import numpy as np
cimport numpy as np
import os, glob, random as rnd
from dicelib.lazytractogram import LazyTractogram
from dicelib.streamline import length as streamline_length
from dicelib.streamline import smooth
from . import ui
from tqdm import trange
from libc.math cimport sqrt


def compute_lenghts( input_tractogram: str, verbose: int=2 ) -> np.ndarray:
    """Compute the lenghts of the streamlines in a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).

    Returns
    -------
    lengths : array of double
        Lengths of all streamlines in the tractogram [in mm]
    """
    if type(verbose) != int or verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )

    #----- iterate over input streamlines -----
    TCK_in = None
    lengths = None
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        if verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )

        lengths = np.empty( n_streamlines, dtype=np.float32 )
        if n_streamlines>0:
            for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False, disable=(verbose in [0,1,3]) ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )

        if verbose and n_streamlines>0:
            ui.INFO( f'min={lengths.min():.3f}   max={lengths.max():.3f}   mean={lengths.mean():.3f}   std={lengths.std():.3f}' )

        return lengths

    except Exception as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()


def info( input_tractogram: str, compute_lengths: bool=False, max_field_length: int=None, verbose: int=2 ):
    """Print some information about a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    compute_lengths : boolean
        Show stats on streamline lenghts (default : False).

    max_field_length : int
        Maximum length allowed for printing a field value (default : all chars)

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).
    """
    if verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

    if max_field_length is not None and max_field_length<25:
        ui.ERROR( '"max_field_length" must be >=25')

    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )

    #----- iterate over input streamlines -----
    TCK_in  = None
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        # print the header
        ui.INFO( 'HEADER content')
        max_len = max([len(k) for k in TCK_in.header.keys()])
        for key, val in TCK_in.header.items():
            if key=='count':
                continue
            if type(val)==str:
                val = [val]
            for v in val:
                if max_field_length is not None and len(v)>max_field_length:
                    v = v[:max_field_length]+ui.hRed+'...'+ui.Reset
                ui.PRINT( ui.hWhite+ '%0*s'%(max_len,key) +ui.Reset+ui.fWhite+ ':  ' + v +ui.Reset )
        if 'count' in TCK_in.header.keys():
            ui.PRINT( ui.hWhite+ '%0*s'%(max_len,'count') +ui.Reset+ui.fWhite+ ':  ' + TCK_in.header['count'] +ui.Reset )
        ui.PRINT( '' )

        # print stats on lengths
        if compute_lengths:
            ui.INFO( 'Streamline lenghts')
            n_streamlines = int( TCK_in.header['count'] )
            if n_streamlines>0:
                lengths = np.empty( n_streamlines, dtype=np.double )
                for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False, disable=(ui.get_verbose() in [0,1,3]) ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading
                    lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                ui.PRINT( f'   {ui.hWhite}min{ui.Reset}{ui.fWhite}={lengths.min():.3f}   {ui.hWhite}max{ui.Reset}{ui.fWhite}={lengths.max():.3f}   {ui.hWhite}mean{ui.Reset}{ui.fWhite}={lengths.mean():.3f}   {ui.hWhite}std{ui.Reset}{ui.fWhite}={lengths.std():.3f}{ui.Reset}' )
            else:
                ui.WARNING( 'The tractogram is empty' )

    except Exception as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()

    if TCK_in.header['count']:
        return TCK_in.header['count']
    else:
        return 0


def filter( input_tractogram: str, output_tractogram: str, minlength: float=None, maxlength: float=None, minweight: float=None, maxweight: float=None, weights_in: str=None, weights_out: str=None, random: float=1.0, verbose: int=2, force: bool=False ):
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
        Scalar file (.txt or .npy) with the input streamline weights.

    weights_out : str
        Scalar file (.txt or .npy) for the output streamline weights.

    random : float
        Probability to keep (randomly) each streamline; this filter is applied after all others (default : 1.0)

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).

    force : boolean
        Force overwriting of the output (default : False).
    """
    if type(verbose) != int or verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

    n_written = 0
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
        weights_in_ext = os.path.splitext(weights_in)[1]
        if weights_in_ext=='.txt':
            w = np.loadtxt( weights_in ).astype(np.float64)
        elif weights_in_ext=='.npy':
            w = np.load( weights_in, allow_pickle=False ).astype(np.float64)
        else:
            ui.ERROR( 'Invalid extension for the weights file' )

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

    if random<=0 or random>1:
        ui.ERROR( '"random" must be in (0,1]' )
    if random!=1:
        ui.INFO( f'Keep streamlines with {random*100:.2f}% probability ' )

    #----- iterate over input streamlines -----
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # check if #(weights)==n_streamlines
        if weights_in is not None and n_streamlines!=w.size:
            ui.ERROR( f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) ' )

        # open the outut file
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

        kept = np.ones( n_streamlines, dtype=bool )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False, disable=(verbose in [0,1,3]) ):
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

            # filter randomly
            if random<1 and rnd.random()>=random:
                kept[i] = False
                continue

            # write streamline to output file
            TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )

        if weights_out is not None and w.size>0:
            if weights_in_ext=='.txt':
                np.savetxt( weights_out, w[kept==True].astype(np.float32), fmt='%.5e' )
            else:
                np.save( weights_out, w[kept==True].astype(np.float32), allow_pickle=False )

        n_written = np.count_nonzero( kept )
        (ui.INFO if n_written>0 else ui.WARNING)( f'{n_written} streamlines in output tractogram' )

    except Exception as e:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( output_tractogram ):
            os.remove( output_tractogram )
        if weights_out is not None and os.path.isfile( weights_out ):
            os.remove( weights_out )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( write_eof=True, count=n_written )


def split( input_tractogram: str, input_assignments: str, output_folder: str='bundles', weights_in: str=None, max_open: int=None, verbose: int=2, force: bool=False ):
    """Split the streamlines in a tractogram according to an assignment file.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to split.

    input_assignments : string
        File containing the streamline assignments (two numbers/row); these can be stored as
        either a simple .txt file or according to the NUMPY format (.npy), which is faster.

    output_folder : string
        Output folder for the splitted tractograms.

    weights_in : string
        Text file with the input streamline weights (one row/streamline). If not None, one individual
        file will be created for each splitted tractogram, using the same filename prefix.

    max_open : integer
        Maximum number of files opened at the same time (default : 90% of SC_OPEN_MAX system variable).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).

    force : boolean
        Force overwriting of the output (default : False).
    """
    if type(verbose) != int or verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if not os.path.isfile(input_assignments):
        ui.ERROR( f'File "{input_assignments}" not found' )
    if not os.path.isdir(output_folder):
        os.mkdir( output_folder )
    else:
        if force:
            for f in glob.glob( os.path.join(output_folder,'*.tck') ):
                os.remove(f)
            for f in glob.glob( os.path.join(output_folder,'*.txt') ):
                os.remove(f)
            for f in glob.glob( os.path.join(output_folder,'*.npy') ):
                os.remove(f)
        else:
            ui.ERROR( 'Output folder already exists, use -f to overwrite' )
    ui.INFO( f'Writing output tractograms to "{output_folder}"' )

    weights_in_ext = None
    if weights_in is not None:
        if not os.path.isfile( weights_in ):
            ui.ERROR( f'File "{weights_in}" not found' )
        weights_in_ext = os.path.splitext(weights_in)[1]
        if weights_in_ext=='.txt':
            w = np.loadtxt( weights_in ).astype(np.float32)
        elif weights_in_ext=='.npy':
            w = np.load( weights_in, allow_pickle=False ).astype(np.float64)
        else:
            ui.ERROR( 'Invalid extension for the weights file' )
        w_idx = np.zeros_like( w, dtype=np.int32 )
        ui.INFO( f'Loaded {w.size} streamline weights' )

    if max_open is None:
        max_open = int( os.sysconf('SC_OPEN_MAX')*0.9 )
    ui.INFO( f'Using {max_open} files open simultaneously' )

    #----- iterate over input streamlines -----
    TCK_in          = None
    TCK_out         = None
    TCK_outs        = {}
    TCK_outs_size   = {}
    if weights_in is not None:
        WEIGHTS_out_idx = {}
    n_written         = 0
    unassigned_count  = 0 
    try:
        # open the tractogram
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # open the assignments
        if os.path.splitext(input_assignments)[1]=='.txt':
            assignments = np.loadtxt( input_assignments, dtype=np.int32 )
        elif os.path.splitext(input_assignments)[1]=='.npy':
            assignments = np.load( input_assignments, allow_pickle=False ).astype(np.int32)
        else:
            ui.ERROR( 'Invalid extension for the assignments file' )
        if assignments.ndim!=2 or assignments.shape[1]!=2:
            print( (assignments.ndim, assignments.shape))
            ui.ERROR( 'Unable to open assignments file' )
        ui.INFO( f'{assignments.shape[0]} assignments in input file' )

        # check if #(assignments)==n_streamlines
        if n_streamlines!=assignments.shape[0]:
            ui.ERROR( f'# of assignments ({assignments.shape[0]}) is different from # of streamlines ({n_streamlines}) ' )
        # check if #(weights)==n_streamlines
        if weights_in is not None and n_streamlines!=w.size:
            ui.ERROR( f'# of weights ({w.size}) is different from # of streamlines ({n_streamlines}) ' )

        # create empty tractograms for unique assignments
        unique_assignments = np.unique(assignments, axis=0)
        for i in range( unique_assignments.shape[0] ):
            if unique_assignments[i,0]==0 or unique_assignments[i,1]==0:
                unassigned_count += 1
                continue
            if unique_assignments[i,0] <= unique_assignments[i,1]:
                key = f'{unique_assignments[i,0]}-{unique_assignments[i,1]}'
            else:
                key = f'{unique_assignments[i,1]}-{unique_assignments[i,0]}'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTractogram( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if weights_in is not None:
                WEIGHTS_out_idx[key] = i+1

        # add key for non-connecting streamlines
        if unassigned_count:
            key = 'unassigned'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTractogram( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if weights_in is not None:
                WEIGHTS_out_idx[key] = 0

        ui.INFO( f'Created {len(TCK_outs)} empty files for output tractograms' )

        #----  iterate over input streamlines  -----
        n_file_open = 0
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False, disable=(verbose in [0,1,3]) ):
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
                    key_to_close = rnd.choice( [k for k,v in TCK_outs.items() if v!=None] )
                    TCK_outs[key_to_close].close( write_eof=False )
                    TCK_outs[key_to_close] = None
                else:
                    n_file_open += 1

                TCK_outs[key] = LazyTractogram( fname, mode='a' )

            # write input streamline to correct output file
            TCK_outs[key].write_streamline( TCK_in.streamline, TCK_in.n_pts )
            TCK_outs_size[key] += 1
            n_written += 1

            # store the index of the corresponding weight
            if weights_in is not None:
                w_idx[i] = WEIGHTS_out_idx[key]

        # create individual weight files for each splitted tractogram
        if weights_in is not None:
            ui.INFO( f'Saving one weights file per bundle' )
            for key in WEIGHTS_out_idx.keys():
                w_bundle = w[ w_idx==WEIGHTS_out_idx[key] ].astype(np.float32)
                if weights_in_ext=='.txt':
                    np.savetxt( os.path.join(output_folder,f'{key}.txt'), w_bundle, fmt='%.5e' )
                else:
                    np.save( os.path.join(output_folder,f'{key}.npy'), w_bundle, allow_pickle=False )

        if unassigned_count:
            ui.INFO( f'{n_written-TCK_outs_size["unassigned"]} connecting, {TCK_outs_size["unassigned"]} non-connecting' )
        else:
            ui.INFO( f'{n_written} connecting, {0} non-connecting' )

    except Exception as e:
        if os.path.isdir(output_folder):
            for key in TCK_outs.keys():
                basename = os.path.join(output_folder,key)
                if os.path.isfile(basename+'.tck'):
                    os.remove(basename+'.tck')
                if weights_in is not None and os.path.isfile(basename+weights_in_ext):
                    os.remove(basename+weights_in_ext)

        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        ui.INFO( 'Closing files' )
        if TCK_in is not None:
            TCK_in.close()
        for key in TCK_outs.keys():
            f = os.path.join(output_folder,f'{key}.tck')
            if not os.path.isfile(f):
                continue
            if TCK_outs[key] is not None:
                TCK_outs[key].close( write_eof=False )
            # Update 'count' and write EOF marker
            tmp = LazyTractogram( f, mode='a' )
            tmp.close( write_eof=True, count=TCK_outs_size[key] )


cpdef spline_smoothing( input_tractogram, output_tractogram=None, control_point_ratio=0.25, segment_len=1.0, verbose=2, force=False ):
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

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 2).

    force : boolean
        Force overwriting of the output (default : False).
    """

    if type(verbose) != int or verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

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
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

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
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False, disable=(verbose in [0,1,3]) ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            smoothed_streamline, n = smooth( TCK_in.streamline, TCK_in.n_pts, control_point_ratio, segment_len )
            TCK_out.write_streamline( smoothed_streamline, n )

    except Exception as e:
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
