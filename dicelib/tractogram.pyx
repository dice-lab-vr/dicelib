#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import numpy as np
cimport numpy as np
import os, glob, random as rnd
from dicelib.lazytractogram import LazyTractogram
from dicelib.streamline import length as streamline_length
from dicelib.streamline import smooth
from . import ui
from dicelib.streamline import apply_smoothing
import nibabel as nib
from libc.math cimport sqrt
from libc.math cimport round as cround


def compute_lenghts( input_tractogram: str, verbose: int=1 ) -> np.ndarray:
    """Compute the lenghts of the streamlines in a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).

    Returns
    -------
    lengths : array of double
        Lengths of all streamlines in the tractogram [in mm]
    """

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
        if n_streamlines>0:
            ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
        else:
            ui.WARNING( 'The tractogram is empty' )

        lengths = np.empty( n_streamlines, dtype=np.float32 )
        if n_streamlines>0:
            with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
                for i in range( n_streamlines ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading

                    lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                    pbar.update()

        if n_streamlines>0:
            ui.INFO( f'min={lengths.min():.3f}   max={lengths.max():.3f}   mean={lengths.mean():.3f}   std={lengths.std():.3f}' )

        return lengths

    except Exception as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()


def info( input_tractogram: str, compute_lengths: bool=False, max_field_length: int=None, verbose: int=4 ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).
    """

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
                with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True ) as pbar:
                    for i in range( n_streamlines ):
                        TCK_in.read_streamline()
                        if TCK_in.n_pts==0:
                            break # no more data, stop reading
                        lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                        pbar.update()
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


def filter( input_tractogram: str, output_tractogram: str, minlength: float=None, maxlength: float=None, minweight: float=None, maxweight: float=None, weights_in: str=None, weights_out: str=None, random: float=1.0, verbose: int=1, force: bool=False ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).

    force : boolean
        Force overwriting of the output (default : False).
    """
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
        with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
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
                pbar.update()

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


def split( input_tractogram: str, input_assignments: str, output_folder: str='bundles', regions: list[int]=[], weights_in: str=None, max_open: int=None, verbose: int=1, force: bool=False ):
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

    regions : list of integers
        If a single integer is provided, only streamlines assigned to that region will be extracted.
        If two integers are provided, only streamlines connecting those two regions will be extracted.

    weights_in : string
        Text file with the input streamline weights (one row/streamline). If not None, one individual
        file will be created for each splitted tractogram, using the same filename prefix.

    max_open : integer
        Maximum number of files opened at the same time (default : 90% of SC_OPEN_MAX system variable).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 1).

    force : boolean
        Force overwriting of the output (default : False).
    """

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
        if len(regions)==0:
            unique_assignments = np.unique(assignments, axis=0)
        elif len(regions)==1:
            assignments.sort()
            unique_assignments = np.unique(assignments[assignments[:,0]==regions[0]], axis=0)
        elif len(regions)==2:
            assignments.sort()
            regions.sort()
            unique_assignments = np.unique(assignments[np.logical_and(assignments[:,0]==regions[0], assignments[:,1]==regions[1])], axis=0)

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
        with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                # skip assignments not in the region list
                elif len(regions)==1 and assignments[i,0]!=regions[0]:
                    continue
                elif len(regions)==2 and (assignments[i,0]!=regions[0] or assignments[i,1]!=regions[1]):
                    continue                
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
                pbar.update()

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


def join( input_list: list[str], output_tractogram: str, weights_list: list[str]=[], weights_out: str=None, verbose: int=1, force: bool=False ):
    """Join different tractograms into a single file.

    Parameters
    ----------
    input_list : list of str
        List of the paths to the files (.tck) to join.

    output_tractogram : string
        Path to the file where to store the resulting tractogram.

    weights_list : list of str
        List of scalar file (.txt or .npy) with the input streamline weights; same order of input_list!

    weights_out : str
        Scalar file (.txt or .npy) for the output streamline weights.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 1).

    force : boolean
        Force overwriting of the output (default : False).
    """

    ui.set_verbose( verbose )

    if len(input_list) < 2:
        ui.ERROR(f'Input list contains less than 2 items')
    for f in input_list:
        if not os.path.isfile( f ):
            ui.ERROR( f'File "{f}" not found' )
    if os.path.isfile(output_tractogram) and not force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )

    ui.INFO( f'Writing output tractogram to "{output_tractogram}"' )

    weights_in_ext = None
    if weights_list:
        if len(input_list) != len(weights_list):
            ui.ERROR( f'Number of weights files is different from number of input trac tograms' )
        for w in weights_list:
            if not os.path.isfile( w ):
                ui.ERROR( f'File "{w}" not found' )
            weights_in_ext = os.path.splitext(w)[1]
            if weights_in_ext not in ['.txt', '.npy']:
                ui.ERROR( f'Invalid extension for the weights file "{w}"' )


    #----- iterate over input files -----
    TCK_in    = None
    TCK_out   = None
    n_written = 0
    weights_tot = np.array([], dtype=np.float32)
    try:
        # open the output file
        TCK_in = LazyTractogram( input_list[0], mode='r' )
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        TCK_in.close()

        with ui.ProgressBar( total=len(input_list), disable=(verbose in [0, 1, 3]) ) as pbar:
            for i,input_tractogram in enumerate(input_list):

                # open the input file
                TCK_in = LazyTractogram( input_tractogram, mode='r' )
                n_streamlines = int( TCK_in.header['count'] )
                if n_streamlines == 0:
                    ui.WARNING( f'NO streamlines found in tractogram {input_tractogram}' )
                else:
                    for s in range( n_streamlines ):
                        TCK_in.read_streamline()
                        if TCK_in.n_pts==0:
                            break # no more data, stop reading
                        TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                        n_written += 1
                TCK_in.close()

                if weights_list:
                    # load weights file
                    weights_in_ext = os.path.splitext(weights_list[i])[1]
                    if weights_in_ext=='.txt':
                        w = np.loadtxt( weights_list[i] ).astype(np.float32)
                    elif weights_in_ext=='.npy':
                        w = np.load( weights_list[i], allow_pickle=False ).astype(np.float64)
                    else:
                        ui.ERROR( 'Invalid extension for the weights file' )
                    # check if #(weights)==n_streamlines
                    if n_streamlines!=w.size:
                        ui.ERROR( f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) in file {input_tractogram}' )
                    # append weights
                    weights_tot = np.append(weights_tot, w)

                pbar.update()

            if weights_out is not None and weights_tot.size>0:
                ui.INFO( f'Writing output weights to    "{weights_out}"' )
                weights_out_ext = os.path.splitext(weights_out)[1]
                if weights_out_ext=='.txt':
                    np.savetxt( weights_out, weights_tot.astype(np.float32), fmt='%.5e' )
                else:
                    np.save( weights_out, weights_tot.astype(np.float32), allow_pickle=False )
                ui.INFO( f'Total output weigths:     {weights_tot.size}' )

        ui.INFO( f'Total output streamlines: {n_written}' )
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


cdef float [:] apply_affine_1pt(float [:] orig_pt, double[::1,:] M, double[:] abc):
    cdef float [:] moved_pt = np.zeros(3, dtype=np.float32)
    moved_pt[0] = float((orig_pt[0]*M[0,0] + orig_pt[1]*M[1,0] + orig_pt[2]*M[2,0]) + abc[0])
    moved_pt[1] = float((orig_pt[0]*M[0,1] + orig_pt[1]*M[1,1] + orig_pt[2]*M[2,1]) + abc[1])
    moved_pt[2] = float((orig_pt[0]*M[0,2] + orig_pt[1]*M[1,2] + orig_pt[2]*M[2,2]) + abc[2])
    return moved_pt

cpdef compute_vect_vers(float [:] p0, float[:] p1):
    cdef float vec_x, vec_y, vec_z = 0
    cdef float ver_x, ver_y, ver_z = 0
    cdef size_t ax = 0
    vec_x = p0[0] - p1[0]
    vec_y = p0[1] - p1[1]
    vec_z = p0[2] - p1[2]
    cdef float s = sqrt( vec_x**2 + vec_y**2 + vec_z**2 )
    ver_x = vec_x / s
    ver_y = vec_y / s
    ver_z = vec_z / s
    return vec_x, vec_y, vec_z, ver_x, ver_y, ver_z

cpdef move_point_to_gm(float[:] point, float vers_x, float vers_y, float vers_z, float step, int chances, int[:,:,::1] gm): 
    cdef bint ok = False
    size_x, size_y, size_z = gm.shape[:3]
    cdef size_t c, a = 0
    cdef int coord_x, coord_y, coord_z = 0
    for c in xrange(chances):
        point[0] = point[0] + vers_x * step
        point[1] = point[1] + vers_y * step
        point[2] = point[2] + vers_z * step
        coord_x = <int>point[0]
        coord_y = <int>point[1]
        coord_z = <int>point[2]
        if coord_x < 0 or coord_y < 0 or coord_z < 0 or coord_x >= size_x or coord_y >= size_y or coord_z >= size_z: # check if I'll moved outside the image space
            break
        if gm[coord_x,coord_y,coord_z] > 0: # I moved in the GM
            ok = True
            break
    return ok, point

def sanitize(input_tractogram: str, gray_matter: str, white_matter: str, output_tractogram: str=None, step: float=0.2, max_dist: float=2, save_connecting_tck: bool=False, verbose: int=2, force: bool=False ):
    """Sanitize stramlines in order to end in the gray matter.
    
    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    gray_matter : string
        Path to the gray matter.

    white_matter : string
        Path to the white matter.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_sanitized' to the input filename.

    step : float = 0.2
        Length of each step done when trying to reach the gray matter [in mm].

    max_dist : float = 2
        Maximum distance tested when trying to reach the gray matter [in mm]. Suggestion: use double (largest) voxel size.
        
    save_connecting_tck : boolean
        Save in output also the tractogram containing only the real connecting streamlines (default : False).
        If True, the file will be created by appending '_only_connecting' to the input filename.
        
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

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_sanitized'+extension

    if os.path.isfile(output_tractogram) and not force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )

    if save_connecting_tck == True :
        basename, extension = os.path.splitext(output_tractogram)
        conn_tractogram = basename+'_only_connecting'+extension
    
    wm_nii = nib.load(white_matter)
    cdef int[:,:,::1] wm = np.ascontiguousarray(wm_nii.get_fdata(), dtype=np.int32)
    wm_header = wm_nii.header
    cdef double [:,::1] wm_affine  = wm_nii.affine
    cdef double [::1,:] M_dir      = wm_affine[:3, :3].T 
    cdef double [:]     abc_dir    = wm_affine[:3, 3]
    cdef double [:,::1] wm_aff_inv = np.linalg.inv(wm_affine) #inverse of affine
    cdef double [::1,:] M_inv      = wm_aff_inv[:3, :3].T 
    cdef double [:]     abc_inv    = wm_aff_inv[:3, 3]
    gm_nii = nib.load(gray_matter)
    cdef int[:,:,::1] gm = np.ascontiguousarray(gm_nii.get_fdata(), dtype=np.int32)
    gm_header = gm_nii.header
    
    if wm.shape[0] != gm.shape[0] or wm.shape[1] != gm.shape[1] or wm.shape[2] != gm.shape[2]:
        ui.ERROR( 'Images have different shapes' )

    if wm_header['pixdim'][1] != gm_header['pixdim'][1] or wm_header['pixdim'][2] != gm_header['pixdim'][2] or wm_header['pixdim'][3] != gm_header['pixdim'][3]:
        ui.ERROR( 'Images have different pixel size' )

    """Modify the streamline in order to reach the GM.
    """
    cdef size_t i, n = 0
    cdef int n_tot   = 0
    cdef int n_in    = 0
    cdef int n_out   = 0
    cdef int n_half  = 0
    TCK_in  = None
    TCK_out = None
    TCK_con = None
    cdef int n_streamlines = 0
    cdef int n_pts_out = 0
    cdef int idx_last  = 0
    cdef int coord_x, coord_y, coord_z = 0
    cdef float[:] tmp  = np.zeros(3, dtype=np.float32)
    cdef float vec_x, vec_y, vec_z = 0
    cdef float ver_x, ver_y, ver_z = 0
    cdef float[:] pt_0 = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_1 = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_2 = np.zeros(3, dtype=np.float32)
    cdef bint extremity = 0 # 0=starting, 1=ending
    cdef bint[:] ok_both  = np.zeros(2, dtype=np.int32) # in GM with starting (0) / ending (1) point?
    cdef bint[:] del_both = np.zeros(2, dtype=np.int32) # have I deleted starting (0) / ending (1) point?

    cdef int chances   = int(round(max_dist / step))
    cdef int chances_f = 0

    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        if n_streamlines == 0:
            ui.ERROR( 'NO streamlines found' )

        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # open the output file
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        if save_connecting_tck==True:
            TCK_con = LazyTractogram( conn_tractogram, mode='w', header=TCK_in.header )

        with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]) ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                n_pts_out = TCK_in.n_pts
                idx_last  = TCK_in.n_pts - 1

                fib = np.asarray(TCK_in.streamline)
                fib = fib[:TCK_in.n_pts, :]
                for n in xrange(3): # move first 3 point at each end
                    fib[n,:] = apply_affine_1pt(fib[n,:], M_inv, abc_inv)
                    fib[idx_last-n,:] = apply_affine_1pt( fib[idx_last-n,:], M_inv, abc_inv)
                fib+=0.5 # move to center

                ok_both  = np.zeros(2, dtype=np.int32)
                del_both = np.zeros(2, dtype=np.int32)

                for extremity in xrange(2):
                    if extremity == 0:
                        coord_x = <int>fib[0,0]
                        coord_y = <int>fib[0,1]
                        coord_z = <int>fib[0,2]
                        pt_0  = fib[0,:]
                        pt_1  = fib[1,:]
                        pt_2  = fib[2,:]
                    else:
                        coord_x = <int>fib[idx_last,0]
                        coord_y = <int>fib[idx_last,1]
                        coord_z = <int>fib[idx_last,2]
                        pt_0  = fib[idx_last,:]
                        pt_1  = fib[idx_last-1,:]
                        pt_2  = fib[idx_last-2,:]
                        
                    if gm[coord_x,coord_y,coord_z]==0: # starting point is outside gm
                        if wm[coord_x,coord_y,coord_z]==1: # starting point is inside wm
                            vec_x, vec_y, vec_z, ver_x, ver_y, ver_z = compute_vect_vers(pt_0, pt_1)
                            tmp = pt_0.copy() # changing starting point, direct
                            ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances, gm)
                            if ok_both[extremity]:
                                if extremity==0: fib[0,:] = tmp.copy()
                                else: fib[idx_last,:] = tmp.copy()
                        if ok_both[extremity] == False: # I used all the possible chances following the direct direction but I have not reached the GM or I stepped outside the image space
                            vec_x, vec_y, vec_z, ver_x, ver_y, ver_z = compute_vect_vers(pt_1, pt_0)
                            tmp = pt_0.copy() # changing starting point, flipped
                            chances_f = int(sqrt( vec_x**2 + vec_y**2 + vec_z**2 ) / step)
                            if chances_f < chances:
                                ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances_f, gm)
                            else:
                                ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances, gm)
                            if ok_both[extremity]:
                                    if extremity==0: fib[0,:] = tmp.copy()
                                    else: fib[idx_last,:] = tmp.copy()
                        if ok_both[extremity] == False: # starting point is outside wm
                            if extremity==0:  # coordinates of second point
                                coord_x = <int>fib[1,0]
                                coord_y = <int>fib[1,1]
                                coord_z = <int>fib[1,2]
                            else: # coordinates of second-to-last point
                                coord_x = <int>fib[idx_last-1,0]
                                coord_y = <int>fib[idx_last-1,1]
                                coord_z = <int>fib[idx_last-1,2]
                            if gm[coord_x,coord_y,coord_z]>0: # second point is inside gm => delete first point
                                ok_both[extremity] = True
                            else: # second point is outside gm
                                if wm[coord_x,coord_y,coord_z]==1: # second point is inside wm
                                    vec_x, vec_y, vec_z, ver_x, ver_y, ver_z = compute_vect_vers(pt_0, pt_2)
                                    tmp = pt_1.copy() # changing starting point, direct
                                    ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances, gm)
                                    if ok_both[extremity]:
                                            if extremity==0: fib[1,:] = tmp.copy()
                                            else: fib[idx_last-1,:] = tmp.copy()
                                else:
                                    vec_x, vec_y, vec_z, ver_x, ver_y, ver_z = compute_vect_vers(pt_2, pt_0)
                                    tmp = pt_1.copy() # changing starting point, flipped
                                    chances_f = int(sqrt( vec_x**2 + vec_y**2 + vec_z**2 ) / step)
                                    if chances_f < chances:
                                        ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances_f, gm)
                                    else:
                                        ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances, gm)
                                    if ok_both[extremity]:
                                            if extremity==0: fib[1,:] = tmp.copy()
                                            else: fib[idx_last-1,:] = tmp.copy()
                            if ok_both[extremity]: # delete first/last point because the second one reaches/is inside GM
                                if extremity==0: fib = np.delete(fib, 0, axis=0)
                                else: fib = np.delete(fib, -1, axis=0)
                                n_pts_out = n_pts_out -1
                                idx_last = idx_last -1
                                del_both[extremity] = True
                    else: # starting point is inside gm
                        ok_both[extremity] = True


                # bring points back to original space
                fib=fib-0.5 # move back to corner
                for n in xrange(2):
                    fib[n,:] = apply_affine_1pt( fib[n,:], M_dir, abc_dir) 
                    fib[idx_last-n,:] = apply_affine_1pt( fib[idx_last-n,:], M_dir, abc_dir) 
                if del_both[0] == False:    
                    fib[2,:] = apply_affine_1pt( fib[2,:], M_dir, abc_dir) 
                if del_both[1] == False:
                    fib[idx_last-2,:] = apply_affine_1pt( fib[idx_last-2,:], M_dir, abc_dir) 

                TCK_out.write_streamline( fib, n_pts_out )
                n_tot += 1

                # count cases
                if ok_both[0] and ok_both[1]:
                    if save_connecting_tck: TCK_con.write_streamline( fib, n_pts_out )
                    n_in += 1
                elif ok_both[0] or ok_both[1]:
                    n_half += 1
                else:
                    n_out += 1
                
                pbar.update()

    except Exception as e:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( output_tractogram ):
            os.remove( output_tractogram )
        if TCK_con is not None:
            TCK_con.close()
        if save_connecting_tck == True :
            if os.path.isfile( conn_tractogram ):
                os.remove( conn_tractogram )
    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( write_eof=True, count=n_tot )
        if TCK_con is not None:
            TCK_con.close( write_eof=True, count=n_in )

    if verbose :
        ui.INFO( f'- Save sanitized tractogram to "{output_tractogram}"' )
        if save_connecting_tck: ui.INFO( f'- Save only connecting streamlines to "{conn_tractogram}"' )
    if verbose :
        ui.INFO( f'    * tot. streamlines: {n_tot}' )
        ui.INFO( f'        + connecting (both ends in GM):          {n_in}' )
        ui.INFO( f'        + half connecting (one ends in GM):      {n_half}' )
        ui.INFO( f'        + non-connecting (both ends outside GM): {n_out}' )


def spline_smoothing_v2( input_tractogram, output_tractogram=None, spline_type='centripetal', epsilon=0.3, segment_len=1.0, verbose=4, force=False ):
    """Smooth each streamline in the input tractogram using Catmull-Rom splines.
    More info at http://algorithmist.net/docs/catmullrom.pdf.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_smooth' to the input filename.

    spline_type : string
        Type of the Catmull-Rom spline: 'centripetal', 'uniform' or 'chordal' (default : 'centripetal').

    epsilon : float
        Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline (default : 0.3).

    segment_len : float
        Sampling resolution of the final streamline after interpolation (default : 1.0).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).

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

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_smooth'+extension

    if spline_type == 'centripetal':
        alpha = 0.5
    elif spline_type == 'chordal':
        alpha = 1.0
    elif spline_type == 'uniform':
        alpha = 0.0
    else:
        ui.ERROR("'spline_type' parameter must be 'centripetal', 'uniform' or 'chordal'")

    if epsilon < 0 :
        raise ValueError( "'epsilon' parameter must be non-negative" )


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
            ui.INFO( f'\t- spline type : {spline_type}')
            ui.INFO( f'\t- segment length : {segment_len:.2f}' )

        # process each streamline
        with ui.ProgressBar( total=n_streamlines ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                smoothed_streamline, n = apply_smoothing(TCK_in.streamline, TCK_in.n_pts, segment_len, epsilon=epsilon, alpha=alpha)
                TCK_out.write_streamline( smoothed_streamline, n )
                pbar.update()

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


cpdef spline_smoothing( input_tractogram, output_tractogram=None, control_point_ratio=0.25, segment_len=1.0, verbose=1, force=False ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).

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
        with ui.ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):

                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                smoothed_streamline, n = smooth( TCK_in.streamline, TCK_in.n_pts, control_point_ratio, segment_len )
                TCK_out.write_streamline( smoothed_streamline, n )
                pbar.update()

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


def recompute_indices(indices, dictionary_kept, verbose=1):
    """Recompute the indices of the streamlines in a tractogram after filtering.

    Parameters
    ----------
    indices : array of integers
        Indices of the streamlines in the original tractogram.

    dictionary_kept : dictionary
        Dictionary of the streamlines kept after filtering.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 4).

    Returns
    -------
    indices_recomputed : array of integers
        Recomputed indices of the streamlines.
    """

    if type(verbose) != int or verbose not in [0,1,2,3,4]:
        ui.ERROR( '"verbose" must be in [0...4]' )
    ui.set_verbose( verbose )

    if verbose==4:
        ui.INFO( 'Recomputing indices' )

    # open indices file and dictionary
    d = np.fromfile(dictionary_kept, dtype=np.uint8)

    idx = np.loadtxt(indices).astype(np.uint32)
    indices_recomputed = []

    # recompute indices
    with ui.ProgressBar( total=idx.size, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
        for i in range( idx.size ):
            #count the number of streamlines before the current one
            n = np.count_nonzero( d[:idx[i]] )

            # check if the current streamline is kept
            if d[idx[i]]==1:
                indices_recomputed.append( n )
            pbar.update()

    return indices_recomputed