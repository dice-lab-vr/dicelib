# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
cimport cython
import warnings
warnings.filterwarnings('ignore', module='dipy')
from libc.math cimport isinf, isnan, NAN, sqrt, atan2, M_PI, round, floor, acos, fmin
from libc.stdio cimport fclose, fgets, fopen, fread, fseek, fwrite, SEEK_CUR, SEEK_END, SEEK_SET
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from libc.string cimport strchr, strlen, strncmp
from libcpp.string cimport string
from dicelib.streamline import apply_smoothing, length as streamline_length, rdp_reduction, smooth, create_streamline_replicas
from dicelib.streamline cimport apply_xform_to_point, set_number_of_points, set_number_of_points_f64
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from time import time
from dicelib.utils import check_params, Dir, File, Num, format_time
from dicelib.connectivity import assign
from dicelib.tsf cimport TrackScalarFile
from scipy.signal import savgol_filter
from scipy.fft import dct
import ast, random as rnd
import os, sys, shutil
import nibabel as nib
import numpy as np
cimport numpy as np
from cython.parallel cimport prange

cdef float[1] NAN1 = {NAN}
cdef float[3] NAN3 = {NAN, NAN, NAN}

logger = setup_logger('tractogram')


@cython.final
cdef class LazyTractogram:
    """Class to 'lazyly' read/write streamlines from tractogram one by one.

    A tractogram can be opened in three different modalities:
    - 'r': reading
    - 'w': writing
    - 'a': appending

    At the moment, only .tck files are supported.
    TODO: complete this description.
    """
    # cdef readonly   str                             filename
    # cdef readonly   str                             suffix
    # cdef readonly   dict                            header
    # cdef readonly   str                             mode
    # cdef readonly   bint                            is_open
    # cdef readonly   float[:,::1]                    streamline
    # cdef readonly   unsigned int                    n_pts
    # cdef            int                             max_points
    # cdef            FILE*                           fp
    # cdef            float*                          buffer
    # cdef            float*                          buffer_ptr
    # cdef            float*                          buffer_end


    def __init__( self, char *filename, char* mode, header=None, unsigned int max_points=3000 ):
        """Initialize the class.

        Parameters
        ----------
        filename : str
            Name of the file containing the tractogram to open.
        mode : str
            Opens the tractogram for reading ('r'), writing ('w') or appending ('a') streamlines.
        header : dictionary, optional
            A dictionary of 'key: value' pairs that define the items in the header; this parameter is only required
            when writing streamlines to disk.
        max_points : unsigned int, default=3000
            The maximum number of points/coordinates allowed for a streamline.
        """
        self.is_open = False
        self.filename = filename
        _, self.suffix = os.path.splitext( filename )
        if self.suffix not in ['.tck']:
            raise ValueError( 'Only ".tck" files are supported for now.' )

        if mode not in ['r', 'w', 'a']:
            raise ValueError( '"mode" must be either "r", "w" or "a"' )
        self.mode = mode

        if mode=='r':
            if max_points<=0:
                raise ValueError( '"max_points" should be positive' )
            self.max_points = max_points
            self.streamline = np.empty( (max_points, 3), dtype=np.float32 )
            self.buffer = <float*> malloc( 3*1000000*sizeof(float) )
        else:
            self.streamline = None
            self.buffer = NULL
        self.n_pts = 0
        self.buffer_ptr = NULL
        self.buffer_end = NULL

        # open the file
        self.fp = fopen( self.filename, ('r+' if self.mode=='a' else self.mode)+'b' )
        if self.fp==NULL:
            raise FileNotFoundError( f'Unable to open file: "{self.filename}"' )

        self.header = {}
        if self.mode=='r':
            # file is open for reading => need to read the header from disk
            self.header.clear()
            self._read_header()
        elif self.mode=='w':
            # file is open for writing => need to write a header to disk
            if header is None:
                header = {'datatype': 'Float32LE', 'timestamp': str(time()), 'count': '0'}
            self._write_header( header )
        else:
            # file is open for appending => move pointer to end
            fseek( self.fp, 0, SEEK_END )

        self.is_open = True


    cpdef int read_streamline( self ) nogil:
        """Read next streamline from the current position in the file.

        For efficiency reasons, multiple streamlines are simultaneously loaded from disk using a buffer.
        The current streamline is stored in the fixed-size numpy array 'self.streamline' and its actual
        length, i.e., number of points/coordinates, is stored in 'self.n_pts'.

        Returns
        -------
        int
            Number of points/coordinates read from disk.
        """
        cdef:
            float* ptr = &self.streamline[0,0]
            int    n_read
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )

        self.n_pts = 0
        while True:
            if self.n_pts>self.max_points:
                raise RuntimeError( f'Problem reading data, streamline seems too long ({self.n_pts}>{self.max_points} points)' )
            if self.buffer_ptr==self.buffer_end: # reached end of buffer, need to reload
                n_read = fread( self.buffer, 4, 3*1000000, self.fp )
                self.buffer_ptr = self.buffer
                self.buffer_end = self.buffer_ptr + n_read
                if n_read < 3:
                    return 0

            # copy coordinate from 'buffer' to 'streamline'
            ptr[0] = self.buffer_ptr[0]
            ptr[1] = self.buffer_ptr[1]
            ptr[2] = self.buffer_ptr[2]
            self.buffer_ptr += 3
            if isnan(ptr[0]) and isnan(ptr[1]) and isnan(ptr[2]):
                break
            if isinf(ptr[0]) and isinf(ptr[1]) and isinf(ptr[2]):
                break
            self.n_pts += 1
            ptr += 3

        return self.n_pts


    cpdef void write_streamline( self, float [:,:] streamline, int n=-1 ) nogil:
        """Write a streamline at the current position in the file.

        Parameters
        ----------
        streamline : Nx3 numpy array
            The streamline data
        n : int
            Writes first n points of the streamline. If n<0 (default), writes all points.
            NB: be careful because, for efficiency, a streamline is represented as a fixed-size array
        """
        if streamline.shape[1]!=3:
            raise RuntimeError( '"streamline" must be a Nx3 array' )
        if n<0:
            n = streamline.shape[0]
        if n==0:
            return

        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode=='r':
            raise RuntimeError( 'File is not open for writing/appending' )

        # write streamline data
        if fwrite( &streamline[0,0], 4, 3*n, self.fp )!=3*n:
            raise IOError( 'Problems writing streamline data to file' )
        # write end-of-streamline signature
        fwrite( NAN3, 4, 3, self.fp )


    cpdef close( self, bint write_eof=True, int count=-1 ):
        """Close the file associated with the tractogram.

        Parameters
        ----------
        write_eof : bool
            Write the EOF marker, i.e. (INF,INF,INF), at the current position (default : True).
            NB: use at your own risk if you know what you are doing.
        count : int
            Update the 'count' field in the header with this value (default : -1, i.e. do not update)
        """
        cdef float inf = float('inf')

        if self.is_open==False:
            return

        if self.mode!='r':
            # write end-of-file marker
            if write_eof:
                fwrite( &inf, 4, 1, self.fp )
                fwrite( &inf, 4, 1, self.fp )
                fwrite( &inf, 4, 1, self.fp )

            # update 'count' in header
            if count>=0:
                if self.mode=='a':
                    # in append mode the header is not read by default
                    self.header.clear()
                    self._read_header()
                self.header['count'] = '%0*d' % (len(self.header['count']), count) # NB: use same number of characters
                self._write_header( self.header )

        self.is_open = False
        fclose( self.fp )
        self.fp = NULL


    cpdef _read_header( self ):
        """Read the header from file.
        After the reading, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to read streamlines.
        """
        cdef char[5000000] line # a field can be max 5MB long
        cdef int           nLines = 0

        if len(self.header) > 0:
            raise RuntimeError( 'Header already read' )

        # check if it's a valid TCK file
        fseek( self.fp, 0, SEEK_SET )
        if fgets(line, sizeof(line), self.fp) == NULL:
            raise IOError( 'Problems reading header from file FIRST LINE' )
        if line.strip() != 'mrtrix tracks':
            raise IOError( f'"{self.filename}" is not a valid TCK file' )

        # parse one line at a time
        while True:
            if nLines>=1000:
                raise RuntimeError( 'Problem parsing the header; too many header lines' )
            if fgets(line, sizeof(line), self.fp) == NULL:
                raise IOError( 'Problems reading header from file' )
            if line.strip() == 'END':
                break
            try:
                key, value = line.strip().split(': ')
            except ValueError:
                raise ValueError('Problem parsing the header; format not valid')
            if key not in self.header:
                self.header[key] = value
            else:
                if type(self.header[key])!=list:
                    self.header[key] = [ self.header[key] ]
                self.header[key].append(value)
            nLines += 1

        # check if the 'count' field is present
        # TODO: fix this, allow working even without it
        if 'count' not in self.header:
            raise RuntimeError( 'Problem parsing the header; field "count" not found' )
        if type(self.header['count'])==list:
            raise RuntimeError( 'Problem parsing the header; field "count" has multiple values' )

        # check if datatype is 'Float32LE'
        if 'datatype' not in self.header:
            raise RuntimeError( 'Problem parsing the header; field "datatype" not found' )
        if type(self.header['datatype'])==list:
            raise RuntimeError( 'Problem parsing the header; field "datatype" has multiple values' )
        if self.header['datatype']!='Float32LE':
            raise RuntimeError( 'Unable to process file, as datatype "Float32LE" is not yet handled' )

        # move file pointer to beginning of binary data
        if 'file' not in self.header:
            raise RuntimeError( 'Problem parsing the header; field "file" not found' )
        if type(self.header['file'])==list:
            raise RuntimeError( 'Problem parsing the header; field "file" has multiple values' )
        fseek(self.fp, int( self.header['file'][2:] ), SEEK_SET)


    cpdef _write_header( self, header ):
        """Write the header to file.
        After writing the header, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to write streamlines.

        Parameters
        ----------
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header.
        """
        cdef string line
        cdef int offset = 18 # accounts for 'mrtrix tracks\n' and 'END\n'

        if header is None or type(header)!=dict:
            raise RuntimeError( 'Provided header is empty or invalid' )

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if 'count' not in header:
            raise RuntimeError( 'Problem parsing the header; field "count" not found' )
        if type(header['count'])==list:
            raise RuntimeError( 'Problem parsing the header; field "count" has multiple values' )

        fseek( self.fp, 0, SEEK_SET )
        line = b'mrtrix tracks\n'
        fwrite( line.c_str(), 1, line.size(), self.fp )

        for key, val in header.items():
            if key=='file':
                continue
            if key=='count':
                val = header['count'] = header['count'].zfill(10) # ensure 10 digits are written

            if type(val)==str:
                val = [val]
            for v in val:
                line = f'{key}: {v}\n'
                fwrite( line.c_str(), 1, line.size(), self.fp )
                offset += line.size()

        line = f'{offset+9:.0f}'
        line = f'file: . {offset+9+line.size():.0f}\n'
        fwrite( line.c_str(), 1, line.size(), self.fp )
        offset += line.size()

        line = b'END\n'
        fwrite( line.c_str(), 1, line.size(), self.fp )

        self.header = header.copy()

        # move file pointer to beginning of binary data
        fseek( self.fp, offset, SEEK_SET )


    cdef void _seek_origin( self, int header_param ) nogil:
        """Move the file pointer to the beginning of the binary data part of the file.
        """
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )
        self.n_pts = 0
        self.buffer_ptr = NULL
        self.buffer_end = NULL
        fseek( self.fp, header_param, SEEK_SET )


    cdef void move_to(self, int n_pts) nogil:
        """Move the file pointer to the specified offset.
        """
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )
        offset = - 3*n_pts*sizeof(float)
        fseek( self.fp, offset, SEEK_CUR )


    def __dealloc__( self ):
        if self.mode=='r':
            free( self.buffer )
        if self.is_open:
            fclose( self.fp )


#---------------------------------------  FUNCTIONS  ---------------------------------------
def compute_lengths( tractogram_filename: str, out_scalars_filename: str=None, force: bool=False , verbose: int=3 ) -> np.ndarray:
    """Compute the lengths [in mm] of each streamline in a tractogram.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    out_scalars_filename : str, default=None
        Path to the file (.txt, .npy) that will contain the estimated lenghts.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    array of float
        Lengths [in mm] of all streamlines in the tractogram.
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Computing lengths of streamlines')

    files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck')]
    if out_scalars_filename is not None:
        files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))
    check_params(files=files, force=force)

    #----- iterate over input streamlines -----
    TCK_in = None
    lengths = None
    try:
        # open the input file
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        if n_streamlines <= 0:
            logger.error('The tractogram is empty')

        lengths = np.empty( n_streamlines, dtype=np.float32 )
        if n_streamlines>0:
            with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
                for i in range( n_streamlines ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading
                    lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                    pbar.update()

        if n_streamlines>0:
            logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
            logger.subinfo(f'min: {lengths.min():.3f}  max: {lengths.max():.3f}  mean: {lengths.mean():.3f}  std: {lengths.std():.3f}', indent_char='*', indent_lvl=1)

        if out_scalars_filename is not None:
            if out_scalars_filename.endswith('.txt'):
                np.savetxt(out_scalars_filename, lengths, fmt='%.4f')
            else:
                np.save(out_scalars_filename, lengths, allow_pickle=False)

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )

    return streamline_length


def info( tractogram_filename: str, max_field_length: int=None, compute_lengths: bool=False, verbose: int=3 ):
    """Print some information about a tractogram.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    max_field_length : int, default=None
        Maximum length allowed for printing the value of each field;
        if not specified, all characters are displayed.
    compute_lengths : boolean, default=False
        Show stats on streamline lengths.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    int
        The number of streamlines in the tractogram.
    """
    set_verbose('tractogram', verbose)

    files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck')]
    nums = None
    if max_field_length is not None:
        nums = [Num(name='max_field_length', value=max_field_length, min_=25)]
    check_params(files=files, nums=nums)

    #----- iterate over input streamlines -----
    TCK_in  = None
    try:
        # open the input file
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )

        # print the header
        max_len = max([len(k) for k in TCK_in.header.keys()])
        for key, val in TCK_in.header.items():
            if key=='count':
                continue
            if type(val)==str:
                val = [val]
            for v in val:
                if max_field_length is not None and len(v)>max_field_length:
                    v = v[:max_field_length] + '...'
                logger.subinfo('%0*s'%(max_len,key) + ':  ' + v)
        if 'count' in TCK_in.header.keys():
            logger.subinfo('%0*s'%(max_len,'count') + ':  ' + TCK_in.header['count'] + '\n')

        # print stats on lengths
        if compute_lengths:
            logger.info('Streamline lengths')
            n_streamlines = int( TCK_in.header['count'] )
            if n_streamlines>0:
                lengths = np.empty( n_streamlines, dtype=np.double )
                with ProgressBar( total=n_streamlines, disable=(verbose < 3), hide_on_exit=True ) as pbar:
                    for i in range( n_streamlines ):
                        TCK_in.read_streamline()
                        if TCK_in.n_pts==0:
                            break # no more data, stop reading
                        lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
                        pbar.update()
                logger.subinfo(f'min: {lengths.min():.3f}  max: {lengths.max():.3f}  mean: {lengths.mean():.3f}  std: {lengths.std():.3f}')
            else:
                logger.error('The tractogram is empty')

    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_in.header['count']:
            return TCK_in.header['count']
        else:
            return 0


def filter( tractogram_filename: str, out_tractogram_filename: str, weights_filename: str=None, minlength: float=None, maxlength: float=None, minweight: float=None, maxweight: float=None, out_weights_filename: str=None, random: float=1.0, scalars_filename: str=None, out_scalars_filename: str=None, force: bool=False, verbose: int=3 ):
    """Filter out the streamlines in a tractogram according to some criteria.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    out_tractogram_filename : str
        Path to the file (.tck) that will contain the filtered tractogram.
    weights_filename : str, optional
        Path to the scalar file (.txt, .npy) containing one weight for each input streamline.
    minlength : float, optional
        Keep streamlines with length [in mm] >= this value.
    maxlength : float, optional
        Keep streamlines with length [in mm] <= this value.
    minweight : float, optional
       Keep streamlines with weight >= this value.
    maxweight : float, optional
        Keep streamlines with weight <= this value.
    out_weights_filename : str, optional
        Path to the scalar file (.txt, .npy) that will contain the weights of the remaining streamlines.
    random : float, deault=1.0
        Percentage of streamlines to keep (randomly): 0=discard all, 1=keep all;
        this filter is applied after all others.
    scalars_filename : str, optional
        Path to the file containing one scalar per streamline (.txt, .npy)
        or one scalar per streamline's coordinate (.tsf).
        This file will be filtered according to the chosen filtering criteria.
    out_scalars_filename : str, optional
        Path to the file (.txt, .npy, .tsf) that will contain scalar information
        of only those streamlines that were not filtered.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Filtering tractogram')

    # check for inconsistency in scalar file perameters
    if (scalars_filename is None) != (out_scalars_filename is None):
        logger.error( 'Input and output scalar files must be either both present or both absent' )

    # check parameters
    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck')
    ]
    if weights_filename is not None:
        files.append(File(name='weights_filename', type_='input', path=weights_filename, ext=['.txt', '.npy']))
    if out_weights_filename is not None:
        files.append(File(name='out_weights_filename', type_='output', path=out_weights_filename, ext=['.txt', '.npy']))
    if scalars_filename is not None:
        files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy', '.tsf']))
    if out_scalars_filename is not None:
        files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy', '.tsf']))

    nums = [Num(name='random', value=random, min_=0.0, max_=1.0, include_min=False)]
    messages = []
    if minlength is not None:
        nums.append(Num(name='minlength', value=minlength, min_=0.0))
        messages.append(f'Keeping streamlines with length >= {minlength}mm')
    if maxlength is not None:
        nums.append(Num(name='maxlength', value=maxlength, min_=0.0))
        messages.append(f'Keeping streamlines with length <= {maxlength}mm')
    if minweight is not None:
        nums.append(Num(name='minweight', value=minweight, min_=0.0))
        messages.append(f'Keeping streamlines with weight >= {minweight}')
    if maxweight is not None:
        nums.append(Num(name='maxweight', value=maxweight, min_=0.0))
        messages.append(f'Keeping streamlines with weight <= {maxweight}')
    if minlength is not None and maxlength is not None and minlength > maxlength:
        logger.error('\'minlength\' must be <= \'maxlength\'')
    if minweight is not None and maxweight is not None and minweight > maxweight:
        logger.error('\'minweight\' must be <= \'maxweight\'')
    if random != 1.0:
        messages.append(f'Randomly keeping {random * 100:.0f}% of the streamlines')
    check_params(files=files, nums=nums, force=force)
    if scalars_filename is not None and (scalars_filename[-4:] != out_scalars_filename[-4:]):
        logger.error( 'Input and output scalar files must have the same format' )

    for msg in messages:
        logger.subinfo(msg, indent_char='*', indent_lvl=1)

    n_written = 0
    TCK_in  = None
    TCK_out = None
    TSF_in  = None
    TSF_out = None
    try:
        # open the input tractogram
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        # create the output tractogram
        TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )
        # load the weights to be used as filtering criterion
        if weights_filename is not None:
            logger.subinfo('Filtering based on streamline weights', indent_char='*', indent_lvl=1)
            if weights_filename.endswith('.txt'):
                w = np.loadtxt(weights_filename).astype(np.float64)
            else:
                w = np.load(weights_filename, allow_pickle=False).astype(np.float64)
            if n_streamlines!=w.size:
                logger.error(f'Number of weights ({w.size}) is different from number of streamlines')
        else:
            w = np.array([])
        # load the additional scalars (if any)
        if scalars_filename is not None:
            if scalars_filename.endswith('.tsf'):
                TSF_in  = TrackScalarFile( scalars_filename, mode='r' )
                TSF_out = TrackScalarFile( out_scalars_filename, mode='w', header=TSF_in.header )

        #----- iterate over input streamlines -----
        with ProgressBar( total=2*n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            kept = np.ones( n_streamlines, dtype=bool )
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
                if weights_filename is not None:
                    if (minweight is not None and w[i]<minweight) or (maxweight is not None and w[i]>maxweight):
                        kept[i] = False
                        continue
                pbar.update()

            if random < 1:
                idx_true = np.where(kept == True)[0]
                discard_choice = np.random.choice( idx_true, int(idx_true.size * (1-random)), replace=False )
                kept[discard_choice] = False

            TCK_in._seek_origin(int(TCK_in.header['file'][2:])) # move position back to data
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if scalars_filename is not None:
                    if scalars_filename.endswith('.tsf'):
                        TSF_in.read_scalars()
                if kept[i]:
                    TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                    if scalars_filename is not None:
                        if scalars_filename.endswith('.tsf'):
                            TSF_out.write_scalars( TSF_in.scalars, TSF_in.n_pts )
                    n_written += 1
                pbar.update()

            if (out_weights_filename is not None) and (w.size > 0):
                if out_weights_filename.endswith('.txt'):
                    np.savetxt(out_weights_filename, w[kept == True].astype(np.float32), fmt='%.5e')
                else:
                    np.save(out_weights_filename, w[kept == True].astype(np.float32), allow_pickle=False)

            if scalars_filename is not None:
                if scalars_filename.endswith('.txt'):
                    scalars_in = np.loadtxt(scalars_filename)
                    np.savetxt(out_scalars_filename, scalars_in[kept == True], fmt='%.5e')
                elif scalars_filename.endswith('.npy'):
                    scalars_in = np.load(scalars_filename, allow_pickle=False)
                    np.save(out_scalars_filename, scalars_in[kept == True], allow_pickle=False)

    except Exception as e:
        if os.path.isfile( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
        if (out_weights_filename is not None) and os.path.isfile( out_weights_filename ):
            os.remove( out_weights_filename )
        if (out_scalars_filename is not None) and os.path.isfile( out_scalars_filename ):
            os.remove( out_scalars_filename )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        logger.subinfo(f'Number of streamlines written: {n_written}', indent_char='*', indent_lvl=1)
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close(write_eof=True, count=n_written )
        if TSF_in is not None:
            TSF_in.close()
        if TSF_out is not None:
            if scalars_filename.endswith('.tsf'):
                TSF_out.close(write_eof=True, count=n_written )
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


def split( tractogram_filename: str, assignments_filename: str, out_folder: str='bundles', prefix: str=None, regions: str=None, scalars_filename: str=None, max_open: int=None, force: bool=False, verbose: int=3, log_list=None ):
    """Split the streamlines in a tractogram according to an assignment file.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to split.
    assignments_filename : str
        Path to the file (.txt, .npy) containing the streamlines' assignments (two numbers/row).
    out_folder : str, default="bundles"
        Output folder for the splitted tractograms.
    prefix : str, optional
        Text to be prepended to the filenames of the output tractograms.
    regions : list of integers, optional
        Only streamlines connecting the provided region(s) will be extracted.
        If not specified, all bundles will be extracted (along with all unassigned streamlines).
        If a single region is provided, all bundles connecting this region with any other will be extracted.
        If a pair of regions is provided using the format "[r1, r2]", only this specific bundle will be extracted.
        If a list of regions is provided using the format "r1, r2, ...", all the possible bundles connecting one of these regions will be extracted.
    scalars_filename : str, optional
        Path to the file (.txt, .npy) containing one scalar for each input streamline (one row/streamline).
        One individual file will be created for each splitted tractogram, using a common prefix.
        If not specified, the streamlines' scalars will not be splitted.
    max_open : int, optional
        Maximum number of concurrent files that can be opened.
        If not specified, the value is automatically set to:
            - on Unix: 90% of half the default system hard limit
            - on Windows: 90% of twice the default system limit
        Else the value exceeds system limits, an attempt is made to adjust it.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info(f'Splitting tractogram')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='assignments_filename', type_='input', path=assignments_filename, ext=['.txt', '.npy'])
    ]
    if scalars_filename is not None:
        files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']))
    dirs = [Dir(name='out_folder', path=out_folder)]
    check_params(files=files, dirs=dirs, force=force)

    if prefix is None:
        prefix = ''

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    def split_regions(input_string):
        try:
            # ast.literal_eval safely parses an input string to a Python literal structure
            return ast.literal_eval(input_string)
        except (SyntaxError, ValueError):
            # Handle the exception if the input string is not a valid Python literal structure
            logger.error('The input string is not a valid Python literal structure.')
            return None

    if not regions==None:
        if not isinstance(split_regions(regions), (list, tuple, int)):
            logger.error('Invalid regions input')
        else:
            regions_str = "[]," + regions
            labels = []
            for r in split_regions(regions_str):
                if r == []:
                    continue
                if isinstance(r, list):
                    if len(r) != 2:
                        logger.error('Invalid regions input')
                labels.append(r)
    else:
        labels = []

    if sys.platform.startswith('win32'):
        import win32file
        limit = win32file._getmaxstdio()
        if max_open is not None and max_open > limit:
            new_limit = int(max_open / 0.9)
            ret = win32file._setmaxstdio(new_limit) # TODO: bug in the library? do not return -1 if not successful (max limit is 2048)
            if ret == -1:
                new_limit = int(limit * 2)
                max_open = int(new_limit * 0.9)
                win32file._setmaxstdio(new_limit)
                warning_msg = f'`max_open` is greater than the system limit, using {max_open} instead'
                logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)
        elif max_open is None:
            new_limit = int(limit * 2)
            max_open = int(new_limit * 0.9)
            win32file._setmaxstdio(new_limit)
    elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        import resource
        limit, limit_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if max_open is not None and max_open > limit:
            new_limit = int(max_open / 0.9)
            if new_limit < limit_hard:
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, limit_hard))
            else:
                new_limit = int(limit_hard * 0.5)
                max_open = int(new_limit * 0.9)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, limit_hard))
                warning_msg = f'`max_open` is greater than the system limit, using {max_open} instead'
                logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)
        elif max_open is None:
            new_limit = int(limit_hard * 0.5)
            max_open = int(new_limit * 0.9)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, limit_hard))
    logger.debug(f'{max_open} files opened simultaneously')

    #----- iterate over input streamlines -----
    TCK_in          = None
    TCK_outs        = {}
    TCK_outs_size   = {}
    if scalars_filename is not None:
        SCALARS_out_idx = {}
    n_written         = 0
    unassigned_count  = 0
    try:
        # open the tractogram
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Output folder: "{out_folder}"', indent_char='*', indent_lvl=1)

        # open the assignments
        if assignments_filename.endswith('.txt'):
            assignments = np.loadtxt(assignments_filename, dtype=np.int32)
        else:
            assignments = np.load(assignments_filename, allow_pickle=False).astype(np.int32)
        if assignments.ndim!=2 or assignments.shape[1]!=2:
            logger.error('Unable to open assignments file')
        logger.subinfo(f'Number of assignments: {assignments.shape[0]}', indent_char='*', indent_lvl=1)

        # open scalar file
        if scalars_filename is not None:
            if scalars_filename.endswith('.txt'):
                w = np.loadtxt(scalars_filename).astype(np.float64)
            else:
                w = np.load(scalars_filename, allow_pickle=False).astype(np.float64)
            w_idx = np.zeros_like(w, dtype=np.int32)
        if scalars_filename is not None:
            logger.subinfo(f'Number of scalars: {w.size}', indent_char='*', indent_lvl=1)

        # check if #(assignments)==n_streamlines
        if n_streamlines!=assignments.shape[0]:
            logger.error(f'Number of assignments ({assignments.shape[0]}) differs from number of streamlines ({n_streamlines})')
        # check if #(scalars)==n_streamlines
        if scalars_filename is not None and n_streamlines!=w.size:
            logger.error(f'Number of scalars ({w.size}) differs from number of streamlines ({n_streamlines})')

        # create empty tractograms for unique assignments
        if len(labels)==0:
            unique_assignments = np.unique(assignments, axis=0)
        else:
            unique_assignments = []
            assignments.sort()
            for r in labels:
                if isinstance(r, int):
                    unique_assignments.extend(np.unique(assignments[assignments[:,0]==r], axis=0))
                    unique_assignments.extend(np.unique(assignments[assignments[:,1]==r], axis=0))
                elif isinstance(r, list):
                    r.sort()
                    unique_assignments.extend(np.unique(assignments[np.logical_and(assignments[:,0]==r[0], assignments[:,1]==r[1])], axis=0))
            # unique_assignments = np.concatenate(unique_assignments, axis=0)
            unique_assignments = np.array(unique_assignments)
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
            pref_key = f'{prefix}{key}'
            tmp = LazyTractogram( os.path.join(out_folder,f'{pref_key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if scalars_filename is not None:
                SCALARS_out_idx[key] = i+1

        # add key for non-connecting streamlines
        if unassigned_count and len(labels)==0:
            key = 'unassigned'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTractogram( os.path.join(out_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if scalars_filename is not None:
                SCALARS_out_idx[key] = 0

        logger.debug(f'Created {len(TCK_outs)} empty files for output tractograms')

        #----  iterate over input streamlines  -----
        n_file_open = 0
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                # skip assignments not in the regions
                if len(labels) > 0:
                    skip = True
                    for r in labels:
                        if isinstance(r, int):
                            if (assignments[i,0]==r):
                                skip = False
                                break
                        elif isinstance(r, list):
                            if (assignments[i,0]==r[0] and assignments[i,1]==r[1]):
                                skip = False
                                break
                    if skip:
                        continue

                    key = f'{assignments[i,0]}-{assignments[i,1]}'

                else:
                    # get the key of the dictionary
                    if assignments[i,0]==0 or assignments[i,1]==0:
                        key = 'unassigned'
                    elif assignments[i,0] <= assignments[i,1]:
                        key = f'{assignments[i,0]}-{assignments[i,1]}'
                    else:
                        key = f'{assignments[i,1]}-{assignments[i,0]}'

                # check if need to open file
                if TCK_outs[key] is None:
                    if key == 'unassigned':
                        pref_key = 'unassigned'
                    else:
                        pref_key = f'{prefix}{key}'
                    fname = os.path.join(out_folder,f'{pref_key}.tck')
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

                # store the index of the corresponding scalar
                if scalars_filename is not None:
                    w_idx[i] = SCALARS_out_idx[key]
                pbar.update()

        # create individual scalar files for each splitted tractogram
        if scalars_filename is not None:
            logger.subinfo(f'Saving one scalar file per bundle', indent_char='*', indent_lvl=1)
            with ProgressBar(disable=verbose < 3, hide_on_exit=True) as pbar:
                for key in SCALARS_out_idx.keys():
                    if key == 'unassigned':
                        pref_key = 'unassigned'
                    else:
                        pref_key = f'{prefix}{key}'
                    w_bundle = w[ w_idx==SCALARS_out_idx[key] ].astype(np.float32)
                    if scalars_filename.endswith('.txt'):
                        np.savetxt( os.path.join(out_folder,f'{pref_key}.txt'), w_bundle, fmt='%.5e' )
                    else:
                        np.save( os.path.join(out_folder,f'{pref_key}.npy'), w_bundle, allow_pickle=False )

        if len(labels)==0:
            if unassigned_count:
                logger.subinfo(f'Connecting streamlines: {n_written-TCK_outs_size["unassigned"]}', indent_char='*', indent_lvl=1)
                logger.subinfo(f'Non-connecting streamlines: {TCK_outs_size["unassigned"]}', indent_char='*', indent_lvl=1)
            else:
                logger.subinfo(f'Connecting streamlines: {n_written}', indent_char='*', indent_lvl=1)

    except Exception as e:
        if os.path.isdir(out_folder):
            for key in TCK_outs.keys():
                pref_key = f'{prefix}{key}'
                basename = os.path.join(out_folder,pref_key)
                if os.path.isfile(basename+'.tck'):
                    os.remove(basename+'.tck')
                in_scalars_ext = os.path.splitext(scalars_filename)[1]
                if scalars_filename is not None and os.path.isfile(basename+in_scalars_ext):
                    os.remove(basename+in_scalars_ext)
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        logger.debug('Closing files')
        with ProgressBar(total=len(TCK_outs), disable=verbose < 3, hide_on_exit=True, subinfo=False) as pbar:
            if TCK_in is not None:
                TCK_in.close()
            for key in TCK_outs.keys():
                if key=='unassigned':
                    pref_key = 'unassigned'
                else:
                    pref_key = f'{prefix}{key}'
                f = os.path.join(out_folder,f'{pref_key}.tck')
                if not os.path.isfile(f):
                    continue
                if TCK_outs[key] is not None:
                    TCK_outs[key].close( write_eof=False )
                # Update 'count' and write EOF marker
                tmp = LazyTractogram( f, mode='a' )
                tmp.close( write_eof=True, count=TCK_outs_size[key] )
                pbar.update()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


def join( tractograms_filenames: list[str], out_tractogram_filename: str, scalars_filenames: list[str]=None, out_scalars_filename: str=None, force: bool=False, verbose: int=3, log_list=None ):
    """Join multiple tractograms into a single file.

    Parameters
    ----------
    tractograms_filenames : list of str
        List of filenames (.tck) of the tractograms to be joined.
    out_tractogram_filename : str
        Path to the file (.tck) that will contain the resulting tractogram.
    scalars_filenames : list of str, optional
        List of paths to the files that contain one scalar for each input streamline
        (.txt, .npy) or one scalar for each point of each input streamline (.tsf).
        These files must follow the same order of the corresponding tractograms.
    out_scalars_filename : str, optional
        Path to the file (.txt, .npy, .tsf) for the output streamline scalars.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Joining multiple tractograms into one')

    if len(tractograms_filenames) < 2:
        logger.error(f'Input list must contain at least 2 files')
    files = [File(name=f'input_tractogram_{i}', type_='input', path=f, ext='.tck') for i, f in enumerate(tractograms_filenames)]
    files.append(File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck'))
    if scalars_filenames is not None:
        if len(tractograms_filenames) != len(scalars_filenames):
            logger.error(f'Number of scalar files is different from number of input tractograms')
        for i, w in enumerate(scalars_filenames):
            files.append(File(name=f'scalars_in_{i}', type_='input', path=w, ext=['.txt', '.npy', '.tsf']))
    if out_scalars_filename is not None:
        files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy', '.tsf']))
    check_params(files=files, force=force)
    if (scalars_filenames is None) != (out_scalars_filename is None):
        logger.error( 'Input and output scalar files must be either both present or both absent' )

    #----- iterate over input files -----
    logger.subinfo(f'Output tractogram: \'{out_tractogram_filename}\'', indent_char='*', indent_lvl=1)
    TCK_in    = None
    TCK_out   = None
    n_written = 0
    try:
        # open the output file
        TCK_in = LazyTractogram( tractograms_filenames[0], mode='r' )
        TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )
        TCK_in.close()

        if out_scalars_filename is not None:
            if out_scalars_filename.endswith('.tsf'):
                TSF_in = TrackScalarFile( scalars_filenames[0], mode='r' )
                TSF_out = TrackScalarFile( out_scalars_filename, mode='w', header=TSF_in.header )
                n_written_tsf = 0
                TSF_in.close()
            else:
                all_scalars = np.array([], dtype=np.float32)


        with ProgressBar( total=len(tractograms_filenames), disable=verbose < 3, hide_on_exit=True) as pbar:
            for i,input_tractogram in enumerate(tractograms_filenames):

                # open the input file
                TCK_in = LazyTractogram( input_tractogram, mode='r' )
                n_streamlines = int( TCK_in.header['count'] )
                if n_streamlines == 0:
                    warning_msg = f'No streamlines found in tractogram {input_tractogram}'
                    logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)
                else:
                    for s in range( n_streamlines ):
                        TCK_in.read_streamline()
                        if TCK_in.n_pts==0:
                            break # no more data, stop reading
                        TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                        n_written += 1
                TCK_in.close()

                if scalars_filenames is not None:
                    # check extension of scalar file
                    if scalars_filenames[i][-4:] != out_scalars_filename[-4:]:
                        logger.error( 'Input and output scalar files must have the same format' )
                    # load scalars file
                    if scalars_filenames[i].endswith('.tsf'):
                        TSF_in = TrackScalarFile( scalars_filenames[i], mode='r' )
                        n_str_tsf = int( TCK_in.header['count'] )
                        # check if n_str_tsf=n_streamlines
                        if n_streamlines!=n_str_tsf:
                            logger.error(f'Number of entries in scalar file ({n_str_tsf}) is different from number of streamlines ({n_streamlines}) in file {input_tractogram}')
                        # write scalars
                        for s in range( n_str_tsf ):
                            TSF_in.read_scalars()
                            if TSF_in.n_pts==0:
                                break # no more data, stop reading
                            TSF_out.write_scalars( TSF_in.scalars, TSF_in.n_pts )
                            n_written_tsf += 1
                        TSF_in.close()
                    else:
                        if scalars_filenames[i].endswith('.txt'):
                            w = np.loadtxt(scalars_filenames[i]).astype(np.float32)
                        else: # npy
                            w = np.load(scalars_filenames[i], allow_pickle=False).astype(np.float64)
                        # check if n_scalars==n_streamlines
                        if n_streamlines!=w.size:
                            logger.error(f'Number of scalars ({w.size}) is different from number of streamlines ({n_streamlines}) in file {input_tractogram}')
                        # append scalars
                        all_scalars = np.append(all_scalars, w)

                pbar.update()

        logger.subinfo(f'Total output streamlines: {n_written}', indent_char='*', indent_lvl=1)
        if out_scalars_filename is not None:
            logger.subinfo(f'Output scalars path: \'{out_scalars_filename}\'', indent_char='*', indent_lvl=1)
            if scalars_filenames[0].endswith('.tsf'):
                TSF_out.close( write_eof=True, count=n_written_tsf )
                logger.subinfo(f'Total output scalars: {n_written_tsf}', indent_char='*', indent_lvl=1)
            else:
                if out_scalars_filename.endswith('.txt'):
                    np.savetxt(out_scalars_filename, all_scalars.astype(np.float32), fmt='%.5e')
                else: # .npy
                    np.save(out_scalars_filename, all_scalars.astype(np.float32), allow_pickle=False)
                logger.subinfo(f'Total output scalars: {all_scalars.size}', indent_char='*', indent_lvl=1)

    except Exception as e:
        if os.path.isfile( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
        if out_scalars_filename is not None and os.path.isfile( out_scalars_filename ):
            os.remove( out_scalars_filename )
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( write_eof=True, count=n_written )
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


def get_indices_of_streamlines( needle_filename: str, haystack_filename: str, out_idx_filename: str=None, force: bool=False, verbose: int=3 ) -> np.ndarray:
    """Finds the indices of a subset of streamlines from a larger tractogram.

    Parameters
    ----------
    needle_filename : str
        Path to the tractogram (.tck) containing the subset of streamlines to find.
    haystack_filename : str
        Path to the tractogram (.tck) containing the full set of streamlines in which to search.
    out_idx_filename : str, optional
        Path to the file (.txt, .npy) that will contain the indices of the streamline that are found.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    array of integers
        Indices of the streamlines from the 'needle' tractogram that where found in 'haystack'.
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Finding indices of streamlines')

    files = [File(name='needle_filename', type_='input', path=needle_filename, ext='.tck'),
             File(name='haystack_filename', type_='input', path=haystack_filename, ext='.tck')]
    if out_idx_filename:
        files.append(File(name='out_idx_filename', type_='output', path=out_idx_filename, ext=['.txt', '.npy']))
    check_params(files=files, force=force)

    TCK_needle = LazyTractogram( needle_filename, mode='r' )
    n_needle = int( TCK_needle.header['count'] )
    logger.subinfo(f'Number of streamlines in needle: {n_needle}', indent_lvl=1, indent_char='*')
    TCK_haystack = LazyTractogram( haystack_filename, mode='r' )
    n_haystack = int( TCK_haystack.header['count'] )
    logger.subinfo(f'Number of streamlines in haystack: {n_haystack}', indent_lvl=1, indent_char='*')

    with ProgressBar(total=n_haystack+n_needle, disable=verbose < 3, hide_on_exit=True) as pbar:
        # hash streamlines in 'haystack' tractogram

        hash_all = np.empty( n_haystack, dtype=int )
        for i in range(n_haystack):
            TCK_haystack.read_streamline()
            n_pts = TCK_haystack.n_pts
            hash_all[i] = hash( np.asarray(TCK_haystack.streamline[:n_pts]).tobytes() )
            pbar.update()
        TCK_haystack.close()

        hash_subset = np.empty( n_needle, dtype=int )
        for i in range(n_needle):
            TCK_needle.read_streamline()
            n_pts = TCK_needle.n_pts
            hash_subset[i] = hash( np.asarray(TCK_needle.streamline[:n_pts]).tobytes() )
            pbar.update()
        TCK_needle.close()

    indices = np.flatnonzero( np.isin( hash_all, hash_subset, assume_unique=True ) )
    logger.subinfo(f'Number of streamlines found: {len(indices)}', indent_lvl=1, indent_char='*')
    # save the indices to file
    if out_idx_filename:
        if out_idx_filename.endswith('.txt'):
            np.savetxt( out_idx_filename, indices, fmt='%d' )
        else:
            np.save( out_idx_filename, indices )

    # return indices of the streamlines that were found
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
    return indices


def sort(tractogram_filename: str, atlas_filename: str, out_tractogram_filename: str=None, distance: float=2.0, scalars_filename: str=None, out_scalars_filename: str=None, tmp_folder: str='tmp_sort', keep_tmp: bool=False, n_threads: int=None, force: bool=False, verbose: int=3 ):
    """Sort the streamlines in a tractogram bundle-by-bundle in lexigraphical order (i.e., 1-1 --> 1-2 --> ... --> 2-2 --> ...).

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to sort.
    atlas_filename : str
        Path to the image (.nii, .nii.gz) containing the labels of the atlas.
    out_tractogram_filename : str, optional
        Path to the tractogram (.tck) that will contain the sorted streamlines. If not specified,
        the output file will be created by appending '_sorted' to the input filename.
    distance : float, default=2.0
        Distance [in mm] to consider in the radial search when computing the assignments.
    scalars_filename : str, optional
        Path to the file (.txt, .npy) containing one scalar for each input streamline.
    out_scalars_filename : str, optional
        Path to the file (.txt, .npy) that will contain the scalars of the sorted streamlines.
    tmp_folder : str, default='tmp_sort'
        Path to the temporary folder used to store the intermediate files.
    keep_tmp : boolean, default=False
        Keep the temporary folder.
    n_threads : int, optional
        How many threads to use in parallel for the computations;
        if not specfied, all available threads will be used.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Sorting streamlines in tractogram')

    # check input files
    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='atlas_filename', type_='input', path=atlas_filename, ext=['.nii', '.nii.gz'])
    ]
    nums = [
        Num(name='distance', value=distance, min_=0.0)
    ]

    if scalars_filename is not None:
        files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']))
        scalars_in_ext = os.path.splitext(scalars_filename)[1]
        if out_scalars_filename is not None:
            scalars_out_ext = os.path.splitext(out_scalars_filename)[1]
            files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))
        else:
            out_scalars_filename = os.path.splitext(scalars_filename)[0]+f'_sorted{scalars_in_ext}'
            scalars_out_ext = scalars_in_ext
            files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))

    if out_tractogram_filename is None:
        out_tractogram_filename = os.path.splitext(tractogram_filename)[0]+'_sorted.tck'
    files.append(File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck'))

    tmp_folder = tmp_folder if tmp_folder is not None else os.path.join(os.getcwd(), 'tmp_sort')
    dirs = [Dir(name='tmp_folder', path=tmp_folder)]
    check_params(files=files, dirs=dirs, nums=nums, force=force)

    tmp_dir_is_created = False
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        tmp_dir_is_created = True

    # compute assignments
    log_list_asgn = []
    ret_subinfo = logger.subinfo('Computing assignments', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo, log_list=log_list_asgn):
        assign(tractogram_filename, atlas_filename, out_assignments_filename=f'{tmp_folder}/fibers_assignment.txt', distance=distance, verbose=1, force=force, n_threads=n_threads, log_list=log_list_asgn)

    # split the tractogram
    log_list_split = []
    ret_subinfo_split = logger.subinfo('Splitting tractogram', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_split, log_list=log_list_split):
        if scalars_filename is not None:
            split(tractogram_filename, f'{tmp_folder}/fibers_assignment.txt', f'{tmp_folder}/bundles', scalars_filename=scalars_filename, verbose=1, force=force, log_list=log_list_split)
        else:
            split(tractogram_filename, f'{tmp_folder}/fibers_assignment.txt', f'{tmp_folder}/bundles', verbose=1, force=force, log_list=log_list_split)
    set_verbose('tractogram', verbose)

    # join the tractograms
    asgn = np.loadtxt( f'{tmp_folder}/fibers_assignment.txt', dtype=np.int32 )
    max_rois = asgn.max()
    log_list_join = []
    ret_subinfo_join = logger.subinfo('Joining bundles in the specific order', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_join, log_list=log_list_join):
        list_all = []
        list_all_scalars = []
        for i in range(max_rois):
            for j in range(i, max_rois):
                path_bundle = f'{tmp_folder}/bundles/{i+1}-{j+1}.tck'
                if os.path.isfile(path_bundle):
                    list_all.append(path_bundle)
                    if scalars_filename is not None:
                        path_scalars = f'{tmp_folder}/bundles/{i+1}-{j+1}{scalars_in_ext}'
                        list_all_scalars.append(path_scalars)
        if scalars_filename is not None:
            join(list_all, out_tractogram_filename, scalars_filenames=list_all_scalars, out_scalars_filename=out_scalars_filename, verbose=1, log_list=log_list_join)
        else:
            join(list_all, out_tractogram_filename, verbose=1, log_list=log_list_join)
    set_verbose('tractogram', verbose)
    if os.path.isfile(f'{tmp_folder}/bundles/unassigned.tck'):
        logger.warning('Some streamlines of the input tractogram are \'non-connecting\'')

    # remove temporary folder/files
    if not keep_tmp:
        shutil.rmtree(f'{tmp_folder}/bundles')
        os.remove(f'{tmp_folder}/fibers_assignment.txt')
        # remove tmp_folder if different from current
        if tmp_dir_is_created:
            shutil.rmtree(tmp_folder)

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


def shuffle(tractogram_filename: str, out_tractogram_filename: str=None, n_tmp_groups: int=100, seed: int=None, scalars_filename: str=None, out_scalars_filename: str=None, tmp_folder: str='tmp_shuffle', keep_tmp: bool=False, force: bool=False , verbose: int=3):
    """Shuffle the streamlines in a tractogram.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to shuffle.
    out_tractogram_filename : str
        Path to the file (.tck) that will contain the shuffled tractogram.
        If not specified, the new file will be created by appending '_shuffled' to the input filename.
    n_tmp_groups : int, default=100
        Number of temporary sub-tractograms to split the streamlines. Each file will contain
        approximately the same number of streamlines, chosen randomly. The final shuffled
        tractogram will be created by concatenating the shuffled groups.
    seed : int, optional
        Uses a specific seed for the random number generator.
    scalars_filename : str, optional
        Path to the file (.txt, .npy) containing one scalar for each input streamline.
    out_scalars_filename : str, optional
        Path to the file (.txt, .npy) that will contain the shuffled streamline scalars.
    tmp_folder : str, default='tmp_shuffle'
        Path to the temporary folder used to store the intermediate files.
    keep_tmp : boolean, default=False
        Keep the temporary folder.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Shuffling streamlines in tractogram')

    # check input files
    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
    ]
    if out_tractogram_filename is None:
        out_tractogram_filename = os.path.splitext(tractogram_filename)[0]+'_shuffled.tck'
    files.append(File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck'))
    nums = [
        Num(name='n_tmp_groups', value=n_tmp_groups, min_=2)
    ]
    if seed is not None:
        nums.append(Num(name='seed', value=seed, min_=0))
    if scalars_filename is not None:
        files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']))
        scalars_in_ext = os.path.splitext(scalars_filename)[1]
        if out_scalars_filename is not None:
            scalars_out_ext = os.path.splitext(out_scalars_filename)[1]
            files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))
        else:
            out_scalars_filename = os.path.splitext(scalars_filename)[0]+f'_shuffled{scalars_in_ext}'
            scalars_out_ext = scalars_in_ext
            files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))
    tmp_folder = tmp_folder if tmp_folder is not None else os.path.join(os.getcwd(), 'tmp_shuffle')
    dirs = [Dir(name='tmp_folder', path=tmp_folder)]
    check_params(files=files, dirs=dirs, nums=nums, force=force)

    tmp_dir_is_created = False
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        tmp_dir_is_created = True

    # create "fake" assignments to split the tractogram
    TCK_in = LazyTractogram( tractogram_filename, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )
    TCK_in.close()
    logger.subinfo(f'Number of input streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
    logger.debug(f'Number of temporary files: {n_tmp_groups}')
    if n_streamlines < n_tmp_groups:
        logger.error(f'Number of temporary groups ({n_tmp_groups}) must be less than the number of streamlines')
    logger.debug(f'Temporary folder: "{tmp_folder}"')
    if seed is not None:
        np.random.seed(seed)
    a = np.repeat(np.arange(1, n_tmp_groups+1), 2, axis=0)
    a = np.reshape(a, (n_tmp_groups, 2))
    n_streamlines_per_group = int( n_streamlines / n_tmp_groups )
    assignments = np.repeat(a, n_streamlines_per_group, axis=0)
    if assignments.shape[0] < n_streamlines:
        assignments = np.append(assignments, np.full((n_streamlines-assignments.shape[0],2), n_tmp_groups, dtype=np.int32), axis=0)
    np.random.shuffle(assignments)
    np.savetxt( f'{tmp_folder}/fake_assignment.txt', assignments, fmt='%d' )

    # split the tractogram
    log_list_split = []
    ret_subinfo_split = logger.subinfo('Splitting into sub-tractograms', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_split, log_list=log_list_split):
        if scalars_filename is not None:
            split(tractogram_filename, f'{tmp_folder}/fake_assignment.txt', f'{tmp_folder}/bundles', scalars_filename=scalars_filename, force=force, verbose=1, log_list=log_list_split)
        else:
            split(tractogram_filename, f'{tmp_folder}/fake_assignment.txt', f'{tmp_folder}/bundles', force=force, verbose=1, log_list=log_list_split)
    set_verbose('tractogram', verbose)

    # join the bundles
    log_list_join = []
    ret_subinfo_join = logger.subinfo('Joining sub-tractograms', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_join, log_list=log_list_join):
        list_all = []
        list_all_scalars = []
        for i in range(1, n_tmp_groups+1):
            path_bundle = f'{tmp_folder}/bundles/{i}-{i}.tck'
            if os.path.isfile(path_bundle):
                list_all.append(path_bundle)
                if scalars_filename is not None:
                    path_scalars = f'{tmp_folder}/bundles/{i}-{i}{scalars_in_ext}'
                    list_all_scalars.append(path_scalars)
        if scalars_filename is not None:
            join(list_all, out_tractogram_filename, scalars_filenames=list_all_scalars, out_scalars_filename=out_scalars_filename, verbose=1, log_list=log_list_join)
        else:
            join(list_all, out_tractogram_filename, verbose=1, log_list=log_list_join)
    set_verbose('tractogram', verbose)
    logger.debug(f'Output tractogram: "{out_tractogram_filename}"')

    # remove temporary folder/files
    if not keep_tmp:
        shutil.rmtree(f'{tmp_folder}/bundles')
        os.remove(f'{tmp_folder}/fake_assignment.txt')
        # remove tmp_folder if different from current
        if tmp_dir_is_created:
            shutil.rmtree(tmp_folder)

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


def sanitize(tractogram_filename: str, gm_filename: str, wm_filename: str, out_tractogram_filename: str, step: float=0.2, max_dist: float=2.0, save_connecting_tck: bool=False, force: bool=False , verbose: int=3):
    """Modify the stramlines to ensure they end inside the gray matter.

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    gm_filename : str
        Path to the image (.nii, .nii.gz) containing the gray-matter labels.
    wm_filename : str
        Path to the image (.niim .nii,.gz) containing the white-matter mask.
    out_tractogram_filename : str
        Path to the tractogram .tck() that will contain the sanitized streamlines.
    step : float, default=0.2
        Length of each step done when trying to reach the gray matter [in mm].
    max_dist : float, default=2
        Maximum distance tested when trying to reach the gray matter [in mm]. Suggestion: use double (largest) voxel size.
    save_connecting_tck : boolean, default=False
        Save in output also the tractogram containing only the real connecting streamlines.
        If True, the file will be created by appending '_only_connecting' to the input filename.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
     """
    def compute_vect_vers(float [:] p0, float[:] p1):
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


    def move_point_to_gm(float[:] point, float vers_x, float vers_y, float vers_z, float step, int chances, int[:,:,::1] gm):
        cdef bint ok = False
        size_x, size_y, size_z = gm.shape[:3]
        cdef size_t c, a = 0
        cdef int coord_x, coord_y, coord_z = 0
        for c in xrange(chances):
            point[0] = point[0] + vers_x * step
            point[1] = point[1] + vers_y * step
            point[2] = point[2] + vers_z * step
            coord_x = <int>round(point[0])
            coord_y = <int>round(point[1])
            coord_z = <int>round(point[2])
            if coord_x < 0 or coord_y < 0 or coord_z < 0 or coord_x >= size_x or coord_y >= size_y or coord_z >= size_z: # check if I'll moved outside the image space
                break
            if gm[coord_x,coord_y,coord_z] > 0: # I moved in the GM
                ok = True
                break
        return ok, point


    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Sanitizing streamlines')

    if out_tractogram_filename is None :
        basename, extension = os.path.splitext(tractogram_filename)
        out_tractogram_filename = basename+'_sanitized'+extension
    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='gm_filename', type_='input', path=gm_filename, ext=['.nii', '.nii.gz']),
        File(name='wm_filename', type_='input', path=wm_filename, ext=['.nii', '.nii.gz']),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck')
    ]
    if save_connecting_tck == True :
        basename, extension = os.path.splitext(out_tractogram_filename)
        conn_tractogram = basename+'_only_connecting'+extension
        files.append(File(name='conn_tractogram', type_='output', path=conn_tractogram, ext='.tck'))
    check_params(files=files, force=force)

    wm_nii = nib.load(wm_filename)
    cdef int[:,:,::1] wm = np.ascontiguousarray(wm_nii.get_fdata(), dtype=np.int32)
    wm_header = wm_nii.header
    cdef double [:,::1] affine  = wm_nii.affine
    cdef double [:,::1] affine_inv = np.linalg.inv(affine) #inverse of affine
    gm_nii = nib.load(gm_filename)
    cdef int[:,:,::1] gm = np.ascontiguousarray(gm_nii.get_fdata(), dtype=np.int32)
    gm_header = gm_nii.header

    if wm.shape[0] != gm.shape[0] or wm.shape[1] != gm.shape[1] or wm.shape[2] != gm.shape[2]:
        logger.error('Images have different shapes')

    if wm_header['pixdim'][1] != gm_header['pixdim'][1] or wm_header['pixdim'][2] != gm_header['pixdim'][2] or wm_header['pixdim'][3] != gm_header['pixdim'][3]:
        logger.error('Images have different pixel size')

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

    cdef int chances   = <int>round(max_dist / step)
    cdef int chances_f = 0
    cdef float [:] moved_pt = np.zeros(3, dtype=np.float32)

    try:
        # open the input file
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        if n_streamlines == 0:
            logger.error('No streamlines found')

        # open the output file
        TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )
        if save_connecting_tck==True:
            TCK_con = LazyTractogram( conn_tractogram, mode='w', header=TCK_in.header )

        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                n_pts_out = TCK_in.n_pts
                idx_last  = TCK_in.n_pts - 1

                fib = np.asarray(TCK_in.streamline)
                fib = fib[:TCK_in.n_pts, :]
                for n in xrange(3): # move first 3 points of the streamline
                    apply_xform_to_point(fib[n,:], affine_inv, moved_pt)
                    fib[n,0] = moved_pt[0]
                    fib[n,1] = moved_pt[1]
                    fib[n,2] = moved_pt[2]
                for n in xrange(3): # move ending 3 points of the streamline
                    apply_xform_to_point( fib[idx_last-n,:], affine_inv, moved_pt)
                    fib[idx_last-n,0] = moved_pt[0]
                    fib[idx_last-n,1] = moved_pt[1]
                    fib[idx_last-n,2] = moved_pt[2]

                ok_both  = np.zeros(2, dtype=np.int32)
                del_both = np.zeros(2, dtype=np.int32)

                for extremity in xrange(2):
                    if extremity == 0:
                        coord_x = <int>round(fib[0,0])
                        coord_y = <int>round(fib[0,1])
                        coord_z = <int>round(fib[0,2])

                        pt_0  = fib[0,:]
                        pt_1  = fib[1,:]
                        pt_2  = fib[2,:]
                    else:
                        coord_x = <int>round(fib[idx_last,0])
                        coord_y = <int>round(fib[idx_last,1])
                        coord_z = <int>round(fib[idx_last,2])

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
                            chances_f = <int>round( sqrt( vec_x**2 + vec_y**2 + vec_z**2 ) / step )
                            if chances_f < chances:
                                ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances_f, gm)
                            else:
                                ok_both[extremity], tmp = move_point_to_gm(tmp, ver_x, ver_y, ver_z, step, chances, gm)
                            if ok_both[extremity]:
                                if extremity==0: fib[0,:] = tmp.copy()
                                else: fib[idx_last,:] = tmp.copy()
                        if ok_both[extremity] == False: # starting point is outside wm
                            if extremity==0:  # coordinates of second point
                                coord_x = <int>round(fib[1,0])
                                coord_y = <int>round(fib[1,1])
                                coord_z = <int>round(fib[1,2])
                            else: # coordinates of second-to-last point
                                coord_x = <int>round(fib[idx_last-1,0])
                                coord_y = <int>round(fib[idx_last-1,1])
                                coord_z = <int>round(fib[idx_last-1,2])
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
                                    chances_f = <int>round( sqrt( vec_x**2 + vec_y**2 + vec_z**2 ) / step )
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
        if os.path.isfile( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
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
        logger.subinfo(f'Sanitized tractogram path: \'{out_tractogram_filename}\'', indent_char='*', indent_lvl=1)
        if save_connecting_tck:
            logger.subinfo(f'Connecting streamlines path: \'{conn_tractogram}\'', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Tot. streamlines: {n_tot}', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Connecting (both ends in GM): {n_in}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Half connecting (one ends in GM): {n_half}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Non-connecting (both ends outside GM): {n_out}', indent_lvl=2, indent_char='-')
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


def smooth_splines( tractogram_filename, out_tractogram_filename, spline_type='centripetal', epsilon=None, n_ctrl_pts=None, n_pts_eval=None, segment_len_eval=None, resample=False, segment_len=None, streamline_pts=None, force=False, verbose=3 ):
    """Smooth the streamlines in a tractogram using Catmull-Rom splines [1].

    The control points of the spline that will approximate a streamline are
    selected using the Ramer-Douglas-Peucker algorithm [2]. Then, these points
    are used to construct a Catmull-Rom spline to approximate its trajectory.

    References:
    [1] https://wikipedia.org/wiki/Catmull–Rom_spline
    [2] https://wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    out_tractogram_filename : str
        Path to the tractogram (.tck) that will contain the smoothed streamlines.
    spline_type : {'centripetal', 'chordal', 'uniform'}, default='centripetal'
        Type of the splines to use.
    epsilon : float, default=0.3
        Distance threshold used by Ramer-Douglas-Peucker algorithm to select the control points of the splines.
    n_ctrl_pts : int, optional
        Use a fixed number of points to select the control points with Ramer-Douglas-Peucker algorithm.
        NB: use either 'epsilon' or 'n_ctrl_pts', not both.
    n_pts_eval : int
        Number of points in which the spline is evaluated. If not specified, the number of points
        is computed using the 'segment_len_eval' parameter.
    segment_len_eval : float, optional
        Segment length used to compute the number of points in which the spline is evaluated;
        this value is computed as the length of the reduced streamline divided by 'segment_len_eval'.
        If not specified, and 'n_pts_eval' is not specified as well, this parameter is set to 0.5.
    resample : boolean, default=False
        Resample the output streamlines to have a constant segment length along the path (see
        'segment_len' and 'streamline_pts' parameters). NB: if False, the output streamlines
        will have more where the curvature is high.
    segment_len : float, optional
        Sampling resolution of the output streamline after interpolation.
    streamline_pts : int, optional
        Number of points of the output streamline after interpolation.
        NB: use either 'segment_len' or 'streamline_pts'.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Smoothing streamlines with splines')

    if n_pts_eval is not None:
        if n_pts_eval < 2:
            logger.error('\'n_pts_eval\' parameter must be greater than 1')
        if segment_len_eval is not None:
            logger.warning('\'segment_len_eval\' parameter will be ignored because \'n_pts_eval\' is set')

    if resample:
        if segment_len is not None and streamline_pts is not None:
            logger.error('Either \'streamline_pts\' or \'segment_len\' must be set, not both.')
        if segment_len is None and streamline_pts is None:
            segment_len = 0.5
            streamline_pts = 0
        else:
            if segment_len is None:
                segment_len = 0
                if streamline_pts < 2:
                    logger.error('\'streamline_pts\' parameter must be greater than 1')
            if streamline_pts is None:
                streamline_pts = 0
        if segment_len_eval is None:
            segment_len_eval = 0.5
    else:
        if segment_len is not None and segment_len != 0:
            logger.warning('\'segment_len\' parameter will be ignored because \'resample\' is set to False')
            segment_len = 0
        if streamline_pts is not None and streamline_pts != 0:
            logger.warning('\'streamline_pts\' parameter will be ignored because \'resample\' is set to False')
            streamline_pts = 0
        if segment_len_eval is None:
            segment_len_eval = 0.5

    if epsilon is not None and n_ctrl_pts is not None:
        logger.error('Either \'epsilon\' or \'n_ctrl_pts\' must be set, not both')
    if epsilon is None and n_ctrl_pts is None:
        epsilon = 0.3
        n_ctrl_pts = 0
    if epsilon is None:
        epsilon = 0
    elif epsilon < 0 :
        logger.error('\'epsilon\' parameter must be non-negative')
    if n_ctrl_pts is None:
        n_ctrl_pts = 0
    elif type(n_ctrl_pts) is not int:
        logger.error(f'\'n_ctrl_pts\'must be an integer data type')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck')
    ]
    check_params(files=files, force=force)

    if spline_type == 'centripetal':
        alpha = 0.5
    elif spline_type == 'chordal':
        alpha = 1.0
    elif spline_type == 'uniform':
        alpha = 0.0
    else:
        logger.error('\'spline_type\' parameter must be \'centripetal\', \'uniform\' or \'chordal\'')
    logger.subinfo(f'Spline type: {spline_type}', indent_lvl=1, indent_char='*')

    try:
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.debug(f'Input tractogram: "{tractogram_filename}"')
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='*')
        TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )

        mb = os.path.getsize( tractogram_filename )/1.0E6
        if mb >= 1E3:
            logger.debug(f'Size: {mb/1.0E3:.2f} GB')
        else:
            logger.debug(f'Size: {mb:.2f} MB')

        if n_ctrl_pts != 0:
            logger.subinfo(f'Number of control points: {n_ctrl_pts}', indent_lvl=1, indent_char='*')
        if epsilon != 0:
            logger.subinfo(f'Number of control points: variable, computed using epsilon={epsilon:.2f}', indent_lvl=1, indent_char='*')

        if resample:
            if segment_len != 0:
                logger.subinfo(f'Resampling in equidistant points with segment length={segment_len:.2f}', indent_lvl=1, indent_char='*')
            if streamline_pts != 0:
                logger.subinfo(f'Resampling in {streamline_pts} equidistant points', indent_lvl=1, indent_char='*')
        else:
            if n_pts_eval is not None:
                logger.subinfo(f'Evaluating the spline in {n_pts_eval} points (not equidistant)', indent_lvl=1, indent_char='*')
            else:
                logger.subinfo(f'Evaluating the spline in a different number of points (not equidistant), depending on the streamline length', indent_lvl=1, indent_char='*')

        logger.debug(f'Output tractogram: "{out_tractogram_filename}"')

        # process each streamline
        n_written = 0
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                smoothed_streamline, n = apply_smoothing(TCK_in.streamline, TCK_in.n_pts, n_pts_final=streamline_pts, segment_len=segment_len, epsilon=epsilon, alpha=alpha, n_pts_red=n_ctrl_pts, n_pts_eval=n_pts_eval, seg_len_eval=segment_len_eval, do_resample=resample)
                TCK_out.write_streamline( smoothed_streamline, n )
                n_written += 1
                pbar.update()

        logger.debug(f'Number of smoothed streamlines: {n_written}')

    except Exception as e:
        if os.path.exists( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( write_eof=True, count=n_written )
        mb = os.path.getsize( out_tractogram_filename )/1.0E6
        if mb >= 1E3:
            logger.debug(f'{mb/1.0E3:.2f} GB')
        else:
            logger.debug(f'{mb:.2f} MB')
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef smooth_savitzky_golay( tractogram_filename, out_tractogram_filename, window=11, polyorder=3, alter_endpoints=False, segment_len=None, force=False, verbose=3 ):
    """Smooth the streamlines in a tractogram using the Savitzky–Golay filter [1].

    References:
    [1] https://en.wikipedia.org/wiki/Savitzky–Golay_filter

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    out_tractogram_filename : str
        Path to the tractogram (.tck) that will contain the smoothed streamlines.
    window : int, default=11
        The length of the filter window.
    polyorder : int, default=3
        The order of the polynomial used to fit the streamline coordinates.
    alter_endpoints : bool, default=False
        Smoothing alters also the endpoints; by default, endpoints are included
        in the filtering window but, after the smoothing, they are restored to
        their original value prior to the filtering to preserve connectivity.
    segment_len : boolean, optional
        If specified, resample the output streamlines to have a constant
        segment length along the path (approximately).
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Smoothing streamlines with the Savitzky–Golay filter')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck')
    ]
    check_params(files=files, force=force)
    if segment_len is not None and segment_len <= 0:
        logger.error('\'segment_len\' parameter must be positive')

    cdef float [::1] lengths = np.empty( 3000, dtype=np.float32 )
    cdef float [:,::1] resampled_streamline = np.empty( (3000, 3), dtype=np.float32 )
    cdef float tot_len
    cdef int n_pts

    try:
        logger.debug(f'Input tractogram: "{tractogram_filename}"')
        logger.debug(f'Output tractogram: "{out_tractogram_filename}"')
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='*')
        TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )
        logger.subinfo(f'Window width: {window}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Polynomial order: {polyorder}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Alter endpoints: {alter_endpoints}', indent_lvl=1, indent_char='*')

        # process each streamline
        with ProgressBar( total=n_streamlines, disable=verbose<3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                smoothed_streamline = savgol_filter(
                    TCK_in.streamline[:TCK_in.n_pts], axis=0,
                    window_length=window, polyorder=polyorder,
                    deriv=0, mode='nearest'
                )
                if alter_endpoints==False:
                    # replace first and last points
                    smoothed_streamline[0,:] = TCK_in.streamline[0,:]
                    smoothed_streamline[TCK_in.n_pts-1,:] = TCK_in.streamline[TCK_in.n_pts-1,:]
                if segment_len is not None:
                    tot_len = streamline_length( smoothed_streamline, TCK_in.n_pts )
                    n_pts = <int>( floor(tot_len/segment_len)+1 )
                    set_number_of_points(smoothed_streamline, TCK_in.n_pts, resampled_streamline, n_pts, lengths)
                    TCK_out.write_streamline( resampled_streamline, n_pts )
                else:
                    TCK_out.write_streamline( smoothed_streamline, TCK_in.n_pts )
                pbar.update()

    except Exception as e:
        if os.path.exists( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


#TODO: describe better what this function does, and its parameters
def recompute_indices(idx_filename, kept_filename, out_idx_filename=None, force=False, verbose=3):
    """Recompute the indices of the streamlines in a tractogram after filtering.

    Parameters
    ----------
    idx_filename : str
        Path to the file (.txt, .npy) containing the indices of the streamlines in the original tractogram.
    kept_filename : dictionary
        Path to the file (.dict) containing the internal dictionary of streamlines kept by COMMIT.
    out_idx_filename : str, optional
        Path to the file (.txt, .npy) that will contain the recomputed indices.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    array of int
        Recomputed indices of the streamlines.
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Recomputing indices')

    files = [
        File(name='idx_filename', type_='input', path=idx_filename, ext=['.txt', '.npy']),
        File(name='kept_filename', type_='input', path=kept_filename, ext='.dict')
    ]
    if out_idx_filename is not None:
        files.append( File(name='out_idx_filename', type_='output', path=out_idx_filename, ext=['.txt', '.npy']) )
    check_params(files=files, force=force)

    # open indices file and dictionary
    d = np.fromfile(kept_filename, dtype=np.uint8)
    if idx_filename.endswith('.txt'):
        idx = np.loadtxt(idx_filename, dtype=np.int32).astype(np.int32)
    else:
        idx = np.load(idx_filename, allow_pickle=False).astype(np.int32)

    indices_recomputed = []
    with ProgressBar( total=idx.size, disable=verbose < 3, hide_on_exit=True) as pbar:
        for i in range( idx.size ):
            # count the number of streamlines before the current one
            n = np.count_nonzero( d[:idx[i]] )
            # check if the current streamline is kept
            if d[idx[i]]==1:
                indices_recomputed.append( n )
            pbar.update()

    if out_idx_filename is not None:
        if out_idx_filename.endswith('.txt'):
            np.savetxt(out_idx_filename, indices_recomputed, fmt='%d')
        else:
            np.save(out_idx_filename, indices_recomputed, allow_pickle=False)

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
    return indices_recomputed


cpdef sample(tractogram_filename, image_filename, out_scalars_filename, mask_filename=None, stat='all', force=False, verbose=3):
    """Sample underlying values of a tractogram along its points from the corresponding image.

    This method does not use interpolation during sampling.

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    image_filename : str
        Path to the image (.nii, .nii.gz) to be sampled.
    out_scalars_filename : str
        Path to the file (.tsf, .txt) that will contain the sampled values.
    mask_filename : str, optional
        Path to the mask (.nii, .nii.gz) to constrain the sampling to a specific region.
    stat : {'all', 'mean', 'median', 'min', 'max'}, default='all'
        Compute a summary statistic on the sampled values;
        if not specified, all values will be saved.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info(f'Sampling scalar values along streamlines')

    if stat not in ['all', 'mean', 'median', 'min', 'max']:
        logger.error('"stat" must be one of [all, mean, median, min, max])')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='image_filename', type_='input', path=image_filename, ext=['.nii','.nii.gz'])
    ]
    if stat=='all':
        files.append( File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.tsf']) )
    else:
        files.append( File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt']) )
    if mask_filename is not None:
        files.append(File(name='mask_filename', type_='input', path=mask_filename, ext=['.nii','.nii.gz']))
    check_params(files=files, force=force)

    # open the scalar image
    niiMAP = nib.load(image_filename)
    niiMAP_img = np.array(niiMAP.get_fdata(), dtype=np.float32)

    # open the mask (if any)
    if mask_filename != None:
        niiMASK_img = np.array(nib.load(mask_filename).get_fdata()>0, dtype=np.uint8)
    else:
        niiMASK_img = np.ones(niiMAP_img.shape, dtype=np.float32)

    cdef float [:,:,::1]            img_view   = np.ascontiguousarray(niiMAP_img).astype(np.float32)
    cdef unsigned char [:,:,::1]    mask_view  = np.ascontiguousarray(niiMASK_img).astype(np.uint8)
    cdef double [:,::1]             affine_inv = np.linalg.inv(niiMAP.affine)
    cdef float [:]                  P          = np.zeros(3, dtype=np.float32)
    cdef float [:]                  values     = np.zeros(3000, dtype=np.float32)
    cdef size_t i, j
    cdef int vx, vy, vz
    TCK_in = None
    TSF_out = None
    try:
        # open the input file
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        pixdim = niiMAP.header['pixdim'] [1:4]
        logger.subinfo(f'Image resolution: {pixdim[0]}x{pixdim[1]}x{pixdim[2]} mm', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Summary statistic: {stat}', indent_char='*', indent_lvl=1)

        # open output file
        cmd = f"dicelib.tractogram.sample {tractogram_filename} {image_filename} {out_scalars_filename}"
        if mask_filename is not None:
            cmd += f" --mask {mask_filename}"
        cmd += f" --stat={stat}"
        if stat == 'all':
            tmp_hdr = TCK_in.header.copy()
            if 'command_history' not in tmp_hdr.keys():
                tmp_hdr['command_history'] = []
            elif type(tmp_hdr['command_history'])==str:
                tmp_hdr['command_history'] = [ tmp_hdr['command_history'] ]
            tmp_hdr['command_history'].append( cmd )
            TSF_out = TrackScalarFile( out_scalars_filename, mode='w', header=tmp_hdr )
            del tmp_hdr
        else:
            file = open(out_scalars_filename,'w')
            file.write( '# '+cmd+'\n' )

        with ProgressBar( total=n_streamlines, disable=verbose<3, hide_on_exit=True) as pbar:
            for i in range(n_streamlines):
                TCK_in.read_streamline()
                for j in range(TCK_in.n_pts):
                    apply_xform_to_point( TCK_in.streamline[j], affine_inv, P )
                    vx = <int>round(P[0])
                    vy = <int>round(P[1])
                    vz = <int>round(P[2])
                    if mask_view[vx, vy, vz] == 0:
                        values[j] = np.nan
                    values[j] = img_view[vx, vy, vz]

                # save sampled values of this streamline to file
                if stat == 'mean':
                    file.write(f'{np.nanmean(values[:TCK_in.n_pts]):.10f}\n')
                elif stat == 'median':
                    file.write(f'{np.nanmedian(values[:TCK_in.n_pts]):.10f}\n')
                elif stat == 'min':
                    file.write(f'{np.nanmin(values[:TCK_in.n_pts]):.10f}\n')
                elif stat == 'max':
                    file.write(f'{np.nanmax(values[:TCK_in.n_pts]):.10f}\n')
                else:
                    TSF_out.write_scalars( values, TCK_in.n_pts )

                pbar.update()

    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if stat=='all':
            if TSF_out is not None:
                TSF_out.close()
        else:
            file.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef resample( tractogram_filename: str, out_tractogram_filename: str, n_pts: int, force: bool=False, verbose: int=3 ):
    """Resample the streamlines in a tractogram to a given number of points.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    out_tractogram_filename : str
        Path to the file (.tck) that will contain the resampled streamlines.
    n_pts : int
        Number of points for resampling the streamlines.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    force : boolean, default=False
        Force overwriting of the output files.
    """
    t0 = time()
    set_verbose('tractogram', verbose)

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck')
    ]
    nums = [Num(name='n_pts', value=n_pts, min_=2)]
    check_params(files=files, nums=nums, force=force)

    cdef float [::1] lengths = np.empty( 3000, dtype=np.float32 )
    cdef float [:,::1] s0 = np.empty( (n_pts, 3), dtype=np.float32 )

    logger.info('Resampling')
    TCK_in = LazyTractogram( tractogram_filename, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )
    logger.subinfo(f'Input tractogram: {tractogram_filename}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Number of points: {n_pts}', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Output tractogram: {out_tractogram_filename}', indent_char='*', indent_lvl=1)

    mb = os.path.getsize( tractogram_filename )/1.0E6
    if mb >= 1E3:
        logger.debug(f'{mb/1.0E3:.2f} GB')
    else:
        logger.debug(f'{mb:.2f} MB')

    # iterate over input streamlines
    TCK_out = LazyTractogram( out_tractogram_filename, mode='w', header=TCK_in.header )
    with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
        for i in range( n_streamlines ):
            TCK_in.read_streamline()
            set_number_of_points(TCK_in.streamline, TCK_in.n_pts, s0, n_pts, lengths)
            TCK_out.write_streamline( s0, n_pts )
            pbar.update()
    TCK_in.close()
    TCK_out.close(write_eof=True, count=n_streamlines)

    mb = os.path.getsize( out_tractogram_filename )/1.0E6
    if mb >= 1E3:
        logger.debug( f'{mb/1.0E3:.2f} GB')
    else:
        logger.debug( f'{mb:.2f} MB')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef save_replicas(input_tractogram: str, output_tractogram: str, blur_core_extent: float, blur_gauss_extent: float, blur_spacing: float=0.25, blur_gauss_min: float=0.1, blur_apply_to=None, save_weights: bool=False, verbose: int=3, force: bool=False ):
    """Save replicas of the input tractogram by applying a Gaussian blur.

    Parameters
    ----------
    input_tractogram : str
        Path to the file (.tck) containing the streamlines to process.
    output_tractogram : str
        Path to the file where to store the output tractogram.
    blur_core_extent: float
        Extent of the core inside which the segments have equal contribution to the central one used by COMMITblur.
    blur_gauss_extent: float
        Extent of the gaussian damping at the border used by COMMITblur.
    blur_spacing : float
        To obtain the blur effect, streamlines are duplicated and organized in a cartesian grid;
        this parameter controls the spacing of the grid in mm (defaut : 0.25).
    blur_gauss_min: float
        Minimum value of the Gaussian to consider when computing the sigma (default : 0.1).
    blur_apply_to: array of bool
        For each input streamline, decide whether blur is applied or not to it (default : None, meaning apply to all).
    save_weights : boolean
        Save the weights of the replicas in the output tractogram (default : False). # TODO: check this output
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose()
    force : boolean, default=False
        Force overwriting of the output files
    """
    t0 = time()
    set_verbose('tractogram', verbose)

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram, ext='.tck')
    ]

    if output_tractogram is not None:
        files.append( File(name='output_tractogram', type_='output', path=output_tractogram, ext='.tck') )
    nums = [
        Num(name='blur_core_extent', value=blur_core_extent, min_=0.0),
        Num(name='blur_gauss_extent', value=blur_gauss_extent, min_=0.0),
        Num(name='blur_spacing', value=blur_spacing, min_=0.0),
        Num(name='blur_gauss_min', value=blur_gauss_min, min_=0.0)
    ]
    check_params(files=files, nums=nums, force=force)
    logger.info('Creating replicas of each streamline in the tractogram')

    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )
    logger.subinfo(f'Input tractogram: {input_tractogram}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'number of streamlines: {n_streamlines}', indent_lvl=2, indent_char='-')

    TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
    n_written = 0

    ####### code from trk2dictionary.pyx #######

    # check for invalid parameters in the blur
    if blur_core_extent < 0 :
        logger.error( 'The extent of the core must be non-negative' )

    if blur_gauss_extent < 0 :
        logger.error( 'The extent of the blur must be non-negative' )

    if blur_gauss_extent > 0 or blur_core_extent > 0:
        if blur_spacing <= 0 :
            logger.error( 'The grid spacing of the blur must be positive' )

    cdef :
        double [:] blurRho
        double [:] blurAngle
        double [:] blurWeights
        cbool [:] blurApplyTo
        int nReplicas
        float blur_sigma
        int i = 0

    if (blur_gauss_extent==0 and blur_core_extent==0) or (blur_spacing==0) :
        nReplicas = 1
        blurRho = np.array( [0.0], np.double )
        blurAngle = np.array( [0.0], np.double )
        blurWeights = np.array( [1], np.double )
    else:
        tmp = np.arange(0,blur_core_extent+blur_gauss_extent+1e-6,blur_spacing)
        tmp = np.concatenate( (tmp,-tmp[1:][::-1]) )
        x, y = np.meshgrid( tmp, tmp )
        r = np.sqrt( x*x + y*y )
        idx = (r <= blur_core_extent+blur_gauss_extent)
        blurRho = r[idx]
        blurAngle = np.arctan2(y,x)[idx]
        nReplicas = blurRho.size

        blurWeights = np.empty( nReplicas, np.double  )
        if blur_gauss_extent == 0 :
            blurWeights[:] = 1.0
        else:
            blur_sigma = blur_gauss_extent / np.sqrt( -2.0 * np.log( blur_gauss_min ) )
            for i in xrange(nReplicas):
                if blurRho[i] <= blur_core_extent :
                    blurWeights[i] = 1.0
                else:
                    blurWeights[i] = np.exp( -(blurRho[i] - blur_core_extent)**2 / (2.0*blur_sigma**2) )

    if nReplicas == 1 :
        logger.subinfo( 'Do not blur streamlines', indent_lvl=2, indent_char='-' )
    else :
        logger.subinfo( 'Blur parameters:', indent_lvl=1, indent_char='*' )
        logger.subinfo( f'core extent  = {blur_core_extent:.3f}', indent_lvl=2, indent_char='-' )
        logger.subinfo( f'gauss extent = {blur_gauss_extent:.3f} (sigma = {blur_sigma:.3f})', indent_lvl=2, indent_char='-' )
        logger.subinfo( f'grid spacing = {blur_spacing:.3f}' , indent_lvl=2, indent_char='-' )
        logger.subinfo( f'weights = [ {np.min(blurWeights):.3f} ... {np.max(blurWeights):.3f} ]', indent_lvl=2, indent_char='-' )
        logger.subinfo( f'n. replicas = {nReplicas:.0f}' , indent_lvl=2, indent_char='-' )

    # check copmpatibility between blurApplyTo and number of streamlines
    if blur_apply_to is None:
        blur_apply_to = np.repeat([True], n_streamlines)
    else :
        if blur_apply_to.size != n_streamlines :
            logger.error( '"blur_apply_to" must have one value per streamline' )
        logger.subinfo( f'{sum(blur_apply_to)} blurred streamlines', indent_lvl=3, indent_char='-' )
    blurApplyTo = blur_apply_to

    ###########################################
    # process each streamline
    with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
        for i in range( n_streamlines ):
            TCK_in.read_streamline()
            nb_pts = TCK_in.n_pts
            str_replicas, pts_replicas = create_streamline_replicas(TCK_in.streamline[:nb_pts], nb_pts, nReplicas, blurRho, blurAngle, blurWeights, blurApplyTo[i])
            for i in range(nReplicas):
                TCK_out.write_streamline( str_replicas[i], pts_replicas[i] )
                n_written += 1
            pbar.update()

    TCK_in.close()
    TCK_out.close( write_eof=True, count=n_written )

    logger.subinfo(f'Output tractogram: {output_tractogram}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'number of streamlines: {n_written}', indent_lvl=2, indent_char='-')

    if save_weights:
        wei_file = output_tractogram.replace('.tck', '_weights.txt')
        logger.subinfo(f'Saving weights: {wei_file}', indent_char='*', indent_lvl=2)
        all_wei = np.tile( blurWeights, n_streamlines )
        np.savetxt( wei_file, all_wei )

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef compute_coherence( tractogram_filename: str, sph_func_filename: str, out_weights_filename: str=None, stat: str='min', percentile: int=5, lobes_filename: str=None, trim: float=0.05, force: bool=False, verbose: int=3 ):
    """Compute the coherence of streamlines with a voxelwise spherical function (e.g. FOD).

    The file containing the spherical functions should follow the MrTrix3 conventions
    for the spherical harmonics; for instance, one can load the FODs estimated with
    MrTrix's dwi2fod command.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    sph_func_filename : str
        Path to the file (.nii, .nii.gz) containing the spherical function against which each streamline is evaluated.
    out_weights_filename : str
        Path to the file (.txt, .npy, .tsf) that will contain the estimated coherence weights.
    stat : {'mean', 'min', 'percentile', 'max', 'all'}, default='min'
        Summary statistic to use once the coherence is computed for all segments of a streamline.
        If 'all' is specified, the coherence of each segment will be saved in a .tsf file;
        otherwise, the summary statistic will be saved in a .txt or .npy file.
        When a .tsf file is produced, each point of a streamline is assigned a weight corresponding
        to the average coherence of the segments centered on that point. For the first and last points,
        the weight is computed using only the following or preceding segment, respectively.
    percentile : int, default=5
        ????
    lobes_filename : string, optional
        Path to the file (.nii, .nii.gz) containing the peaks that identify the lobes of the spherical functions, which
        will be used to normalize the local coherence by the value of the corresponding lobe.
    trim : float, default=0.05
        Percentage of segments to skip at each extremity.
        Note: if 'stat' is set to 'all', only the segments that are not trimmed will be saved in the output file and the
        points that are extremities of the trimmed segments will be assigned a weight of -1 by default.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    array of float
        The estimate coherence weights for all input streamlines.
    """
    import amico.lut
    from dipy.reconst.shm import real_sh_tournier
    #TODO: remove DIPY dependency (currently, required for only 1 function)
    cdef float [::1] w = np.empty(10000, dtype=np.float32) #NOTE: assume max length of a streamline = 10000
    cdef float [:] p1 = np.empty(3, dtype=np.float32)
    cdef float [:] p2 = np.empty(3, dtype=np.float32)
    cdef float [:] dir = np.empty(3, dtype=np.float32)
    cdef short [:] htable
    cdef float [:,:,:,::1] niiSF_img
    cdef float [:,:,:,::1] niiPEAKS_img
    cdef float [:,::1] sh_basis
    cdef float [:,::1] dirs_angles = np.zeros((500,500), dtype=np.float32)
    cdef float [:] dirs_angles_voxel
    cdef float [::1] sf_voxel = np.empty(500, dtype=np.float32)
    cdef float [:] coherence
    cdef float [:] coherence_tsf = np.empty(10000, dtype=np.float32)
    cdef int [:] peaks_idx
    cdef double [:,::1] affine_inv
    cdef LazyTractogram TCK_in = None
    cdef TrackScalarFile TSF_out = None
    cdef int ox, oy, o, o2, trim_offset, n
    cdef int vx, vy, vz, i, j, k, n_peaks=0, peaks_found
    cdef float sf_val1, sf_val2
    cdef float *ptr1
    cdef float *ptr2

    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Computing coherence')

    if trim<0 or trim>=0.5:
        logger.error('"trim" must be in [0..0.5)')
    if stat not in ['mean','min','percentile','max','all']:
        logger.error('"stat" must be one of [mean, min, percentile, max, all]')
    if stat=='percentile' and (percentile<0 or percentile>100):
        logger.error('"percentile" must be an integer in the range [0..100]')

    files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck'])]
    files.append(File(name='sph_func_filename', type_='input', path=sph_func_filename, ext=['.nii', '.nii.gz']))
    if out_weights_filename is not None:
        if stat == 'all':
            files.append(File(name='out_weights_filename', type_='output', path=out_weights_filename, ext=['.tsf']))
        else:
            files.append(File(name='out_weights_filename', type_='output', path=out_weights_filename, ext=['.txt', '.npy']))
    if lobes_filename is not None:
        files.append(File(name='lobes_filename', type_='input', path=lobes_filename, ext=['.nii', '.nii.gz']))
    check_params(files=files, force=force)

    try:
        # open tractogram
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        if stat == 'all':
            TSF_out = TrackScalarFile( out_weights_filename, mode='w', header=TCK_in.header )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        if n_streamlines <= 0:
            logger.error('The tractogram is empty')
        logger.subinfo(f'Trimming {trim*100:.1f}% of segments at each extremity', indent_char='*', indent_lvl=1)

        # open spherical functions
        niiSF = nib.load( sph_func_filename )
        niiSF_img = np.ascontiguousarray(niiSF.get_fdata(), dtype=np.float32)
        n_sh_coeff = niiSF_img.shape[3]
        lmax = (-3.0 + sqrt(1+8*n_sh_coeff)) / 2
        if not lmax.is_integer() :
            logger.error( f'The number of coefficients ({n_sh_coeff}) is not compatible with any SH basis' )
        lmax = int(lmax)
        affine_inv  = np.linalg.inv(niiSF.affine)
        logger.subinfo(f'Spherical functions order: {lmax:.0f}', indent_char='*', indent_lvl=1)

        # construct the SH basis to sample the spherical function
        # (using the 500 directions/hash table used internally by COMMIT/AMICO)
        logger.debug( 'Computing SH basis' )
        dirs  = amico.lut.load_directions( 500 )
        logger.debug( f'directions: {dirs.shape[0]}x{dirs.shape[1]}'  )
        htable = amico.lut.load_precomputed_hash_table( 500 )
        logger.debug( f'hash table: {htable.shape[0]}x1 [min={np.min(htable)}, max={np.max(htable)}]'  )
        theta = np.zeros(dirs.shape[0])
        phi = np.zeros(dirs.shape[0])
        for i in range(theta.size):
            phi[i] = atan2(dirs[i,1], dirs[i,0])
            theta[i] = atan2(sqrt(dirs[i,0]*dirs[i,0]+dirs[i,1]*dirs[i,1]), dirs[i,2])
            dirs[i,:] /= np.linalg.norm(dirs[i,:]) # normalize for later computation
        tmp, _, _ = real_sh_tournier(lmax, theta, phi)
        sh_basis = np.asarray(tmp, dtype=np.float32)
        del theta, phi, tmp

        # open lobes for normalization
        niiPEAKS = None
        if lobes_filename is not None:
            logger.subinfo('Normalizing by lobes', indent_char='*', indent_lvl=1)
            niiPEAKS = nib.load( lobes_filename )
            niiPEAKS_img = np.ascontiguousarray(niiPEAKS.get_fdata(), dtype=np.float32)
            logger.debug(f'Peaks of the lobes: {niiPEAKS.shape[0]}x{niiPEAKS.shape[1]}x{niiPEAKS.shape[2]}x{niiPEAKS.shape[3]}')
            if niiPEAKS.shape[:3] != niiSF.shape[:3]:
                logger.error( f'The shape of the PEAKS dataset is not compatible with the SPHERICAL FUNCTIONS' )
            if niiPEAKS.shape[3] % 3:
                logger.error( 'PEAKS dataset must have 3*k volumes' )
            n_peaks = niiPEAKS.shape[3]/3
            dirs_angles_voxel = np.zeros(n_peaks, dtype=np.float32)
            logger.debug( 'Computing angles between 500 directions' )
            for i in range(500):
                for j in range(i+1,500):
                    dirs_angles[i,j] = acos(dirs[i,0]*dirs[j,0]+dirs[i,1]*dirs[j,1]+dirs[i,2]*dirs[j,2])
                    dirs_angles[j,i] = dirs_angles[i,j]
            peaks_idx = np.zeros(n_peaks, dtype=np.int32)
        del dirs

        if stat!='percentile':
            logger.subinfo(f'Summary statistic along streamlines: "{stat}"', indent_char='*', indent_lvl=1)
        else:
            logger.subinfo(f'Summary statistic along streamlines: "{percentile}-th percentile"', indent_char='*', indent_lvl=1)

        #----- process every streamline -----
        coherence = np.zeros( n_streamlines, dtype=np.float32 )
        if n_streamlines>0:
            with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
                for i in range( n_streamlines ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading
                    if TCK_in.n_pts>10000:
                        logger.error( f'The streamline {i} contains too many points ({TCK_in.n_pts})' )

                    trim_offset = <int>round((TCK_in.n_pts-1)*trim) # skip 'trim' percent of segments
                    if TCK_in.n_pts - trim_offset*2 <=0 :
                        logger.warning( f'"trim" too high, streamline {i} is empty; coherence set to 0' )
                        coherence[i] = 0
                        continue

                    apply_xform_to_point(TCK_in.streamline[trim_offset], affine_inv, p1)
                    n = 0
                    for j in range(trim_offset+1,TCK_in.n_pts-trim_offset):
                        # get direction of current segment
                        apply_xform_to_point(TCK_in.streamline[j], affine_inv, p2)

                        # compute polar angles (NB: hash tables cover half sphere)
                        dir[1] = p2[1]-p1[1]
                        if dir[1] < 0:
                            dir[1] = -dir[1]
                            dir[0] = p1[0]-p2[0]
                            dir[2] = p1[2]-p2[2]
                        else:
                            dir[0] = p2[0]-p1[0]
                            dir[2] = p2[2]-p1[2]

                        # round to the closest direction among the canonical 500 internally used by AMICO/COMMIT
                        ox = <int>round(atan2( sqrt(dir[0]*dir[0]+dir[1]*dir[1]), dir[2] )/M_PI*180.0)
                        oy = <int>round(atan2( dir[1], dir[0] )/M_PI*180.0)
                        o = htable[ox*181+oy]

                        # evaluate the SF along this direction (i.e. sh_basis[o,:] @ niiSF_img[vx,vy,vz,:])
                        vx = <int>round(0.5*(p2[0]+p1[0]))
                        vy = <int>round(0.5*(p2[1]+p1[1]))
                        vz = <int>round(0.5*(p2[2]+p1[2]))
                        ptr1 = &niiSF_img[vx,vy,vz,0]
                        ptr2 = &sh_basis[o,0]
                        sf_val1 = 0
                        for k in range(n_sh_coeff):
                            sf_val1 += ptr1[k]*ptr2[k]
                        if sf_val1 < 0.0:
                            sf_val1 = 0.0

                        # normalize by corresponding lobe
                        if n_peaks > 0:
                            ptr2 = &niiPEAKS_img[vx,vy,vz,0]
                            peaks_found = 0
                            for k in range(n_peaks):
                                if isnan(ptr2[0]):# or (ptr2[0]==0 and ptr2[1]==0 and ptr2[2]==0):
                                    break
                                peaks_found += 1

                                # compute polar angles (NB: hash tables cover half sphere)
                                if ptr2[1] > 0:
                                    ox = <int>round(atan2( sqrt(ptr2[0]*ptr2[0]+ptr2[1]*ptr2[1]), ptr2[2] )/M_PI*180.0)
                                    oy = <int>round(atan2( ptr2[1], ptr2[0] )/M_PI*180.0)
                                else:
                                    ox = <int>round(atan2( sqrt(ptr2[0]*ptr2[0]+ptr2[1]*ptr2[1]), -ptr2[2] )/M_PI*180.0)
                                    oy = <int>round(atan2( -ptr2[1], -ptr2[0] )/M_PI*180.0)
                                o2 = htable[ox*181+oy]
                                if o<0 or o>=500 or o2<0 or o2>=500:
                                    logger.error( f'This should not happen: o={o} o2={o2}' )
                                dirs_angles_voxel[k] = dirs_angles[o,o2] # angle between segment and k-th lobe
                                peaks_idx[k] = o2
                                ptr2 += 3

                            if peaks_found>0:
                                k = peaks_idx[ np.argmin(dirs_angles_voxel[:peaks_found]) ]
                                ptr2 = &sh_basis[k,0]
                                sf_val2 = 0
                                for k in range(n_sh_coeff):
                                    sf_val2 += ptr1[k]*ptr2[k]
                                if sf_val2 < 0.0:
                                    sf_val2 = 0.0
                                if sf_val2 > sf_val1:
                                    sf_val1 /= sf_val2
                                else:
                                    sf_val1 = 1.0 # crop top 1
                        w[n] = sf_val1 if sf_val1>0 else 0
                        n += 1

                        # process next coordinate along streamline
                        p1[0] = p2[0]
                        p1[1] = p2[1]
                        p1[2] = p2[2]

                    if stat=='mean':
                        coherence[i] = np.mean(w[:n])
                    elif stat=='min':
                        coherence[i] = np.min(w[:n])
                    elif stat=='percentile':
                        coherence[i] = np.percentile(w[:n], percentile)
                    elif stat=='max':
                        coherence[i] = np.max(w[:n])
                    elif stat=='all':
                        coherence_tsf[:trim_offset] = -1
                        coherence_tsf[trim_offset] = w[0]
                        for j in range(n-1):
                            coherence_tsf[trim_offset+j+1] = (w[j]+w[j+1])/2.0
                        coherence_tsf[TCK_in.n_pts-trim_offset-1] = w[n-1]
                        coherence_tsf[TCK_in.n_pts-trim_offset:] = -1
                        TSF_out.write_scalars( coherence_tsf, TCK_in.n_pts )

                    pbar.update()

            if stat != 'all':
                logger.subinfo(f'Estimated coherence:  {np.mean(coherence):.3f} ± {np.std(coherence):.3f} [min={np.min(coherence):.3f}, max={np.max(coherence):.3f}]', indent_char='*', indent_lvl=1)

        if out_weights_filename is not None:
            if stat == 'all':
                if TSF_out is not None:
                    TSF_out.close()
            elif out_weights_filename.endswith('.txt'):
                np.savetxt(out_weights_filename, coherence, fmt='%.4f')
            else:
                np.save(out_weights_filename, coherence, allow_pickle=False)

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )
        return coherence


cpdef compute_tdi( tractogram_filename: str, ref_image_filename: str, out_map_filename: str, force: bool=False, verbose: int=3 ):
    """Compute the voxelwise TDI map from a tractogram.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    ref_image_filename : str
        Path to the reference image (.nii, .nii.gz) to infer geometry/orientation.
    out_map_filename : str
        Path to the file (.nii, .nii.gz) that will contain the estimated TDI map.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Computing TDI')

    cdef float [:] p1 = np.zeros(3, dtype=np.float32)
    cdef float [:] p2 = np.zeros(3, dtype=np.float32)
    cdef float [:] P
    cdef double [:,::1] affine_inv
    cdef float [:,:,::1] niiTDI_img
    cdef int vx, vy, vz

    files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck'])]
    files.append(File(name='ref_image_filename', type_='input', path=ref_image_filename, ext=['.nii','.nii.gz']))
    if out_map_filename is not None:
        files.append(File(name='out_map_filename', type_='output', path=out_map_filename, ext=['.nii','.nii.gz']))
    check_params(files=files, force=force)

    #----- iterate over input streamlines -----
    TCK_in = None
    try:
        # open tractogram
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        if n_streamlines <= 0:
            logger.error('The tractogram is empty')

        # open reference image
        niiREF = nib.load( ref_image_filename )
        logger.subinfo(f'Reference image: {niiREF.shape[0]}x{niiREF.shape[1]}x{niiREF.shape[2]}', indent_char='*', indent_lvl=1)
        affine_inv  = np.linalg.inv(niiREF.affine)

        # process every streamline
        niiTDI_img = np.zeros( niiREF.shape[:3], dtype=np.float32 )
        if n_streamlines>0:
            with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
                for i in range( n_streamlines ):
                    TCK_in.read_streamline()
                    if TCK_in.n_pts==0:
                        break # no more data, stop reading

                    P = TCK_in.streamline[0]
                    apply_xform_to_point(P, affine_inv, p1)
                    n = 0
                    for j in range(TCK_in.n_pts):
                        P = TCK_in.streamline[j]
                        apply_xform_to_point(P, affine_inv, p2)
                        # assign the whole segment length to the voxel of its centrois
                        #FIXME: allow better computation of segments contributions in voxels
                        vx = <int>round(0.5*(p2[0]+p1[0]))
                        vy = <int>round(0.5*(p2[1]+p1[1]))
                        vz = <int>round(0.5*(p2[2]+p1[2]))
                        niiTDI_img[vx,vy,vz] += sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
                        # update point
                        p1[0] = p2[0]
                        p1[1] = p2[1]
                        p1[2] = p2[2]
                    pbar.update()
            logger.subinfo(f'Estimated values:  min={np.min(niiTDI_img):.3f}  max={np.max(niiTDI_img):.3f}  mean={np.mean(niiTDI_img):.3f}  std={np.std(niiTDI_img):.3f}', indent_char='*', indent_lvl=1)
        nib.Nifti1Image( niiTDI_img, niiREF.affine ).to_filename( out_map_filename )

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


cdef inline unsigned char _is_within_distance(const float[:,:,::1] streamlines, int idx, const float[:,:,::1] streamlines_bundle, float thr, int n_pts_full) noexcept nogil:
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t n_dct = streamlines.shape[1]
        Py_ssize_t n_str_bundle = streamlines_bundle.shape[0]
        float dx, dy, dz, tmp1
        float dist_direct, dist_flipped
        float thr_scaled = thr * n_pts_full

    for i in range(n_str_bundle):
        # ---- explicit first iteration ----
        dx = streamlines[idx,0,0] - streamlines_bundle[i,0,0]
        dy = streamlines[idx,0,1] - streamlines_bundle[i,0,1]
        dz = streamlines[idx,0,2] - streamlines_bundle[i,0,2]
        dist_direct = dist_flipped = dx*dx + dy*dy + dz*dz
        if dist_direct > thr_scaled:
            continue # too far, process next streamline

        # odd = 1
        dx  = streamlines[idx,1,0] - streamlines_bundle[i,1,0]
        dy  = streamlines[idx,1,1] - streamlines_bundle[i,1,1]
        dz  = streamlines[idx,1,2] - streamlines_bundle[i,1,2]
        dist_direct += dx*dx + dy*dy + dz*dz
        dx = streamlines[idx,1,0] + streamlines_bundle[i,1,0]
        dy = streamlines[idx,1,1] + streamlines_bundle[i,1,1]
        dz = streamlines[idx,1,2] + streamlines_bundle[i,1,2]
        dist_flipped += dx*dx + dy*dy + dz*dz
        if fmin(dist_direct, dist_flipped) > thr_scaled:
            continue # too far, process next streamline

        for j in range(2, n_dct, 2):
            # even = [2, 4, 6, ...]
            dx = streamlines[idx,j,0] - streamlines_bundle[i,j,0]
            dy = streamlines[idx,j,1] - streamlines_bundle[i,j,1]
            dz = streamlines[idx,j,2] - streamlines_bundle[i,j,2]
            tmp1 = dx*dx + dy*dy + dz*dz
            dist_direct  += tmp1
            dist_flipped += tmp1
            # if fmin(dist_direct, dist_flipped) > thr_scaled:
                # break # too far, process next streamline

            # odd = [3, 5, 7, ...]
            k = j+1
            dx  = streamlines[idx,k,0] - streamlines_bundle[i,k,0]
            dy  = streamlines[idx,k,1] - streamlines_bundle[i,k,1]
            dz  = streamlines[idx,k,2] - streamlines_bundle[i,k,2]
            dist_direct  += dx*dx + dy*dy + dz*dz
            dx = streamlines[idx,k,0] + streamlines_bundle[i,k,0]
            dy = streamlines[idx,k,1] + streamlines_bundle[i,k,1]
            dz = streamlines[idx,k,2] + streamlines_bundle[i,k,2]
            dist_flipped += dx*dx + dy*dy + dz*dz
            # if fmin(dist_direct, dist_flipped) > thr_scaled:
                # break # too far, process next streamline

        if fmin(dist_direct, dist_flipped) <= thr_scaled:
            return 1 # streamline is close enough to one of the streamlines in the bundle

    return 0 # streamline is not close enough to any of the streamlines in the bundle


cpdef recognize_streamlines( tractogram_filename: str, bundle_filenames: list[str], out_folder: str="output", thr: float=36.0, n_sub: int=12, n_dct: int=6, suffix: str="", n_threads: int=None, force: bool=False, verbose: int=3 ):
    """Search and recognize streamlines of a tractogram that are close (up to a given threshold) to those of a second tractogram.

    Parameters
    ----------
    tractogram_filename : str
        Path to the file (.tck) containing the streamlines to process.
    bundle_filenames : str
        List of filenames (.tck) of the tractograms containing the streamlines
        to compare to; wildcard characters are allowed, e.g. "folder/*.tck"
    thr : float
        Maximum ASED distance for a streamline to be recognized.
    out_folder : str
        Path to the folder that will contain the recognied streamlines.
    n_sub : int, default=12
        Number of points for streamline resampling.
    n_dct : int, default=6
        Number of DCT coefficients for calculating distances.
    suffix : string, optional
        String to append to the filename of the recognied bundle(s).
    n_threads : int, optional
        How many threads to use for parallel computations;
        if not specfied, all available cores will be used.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().

    Returns
    -------
    array of bool
        A bollean value for each input streamline: 1=recognized, 0=otherwise.
    """
    cdef:
        size_t i, j, k, l
        unsigned char [::1] is_found
        LazyTractogram TCK_in = None, TCK_out = None
        int n_streamlines, n_streamlines_bundle, _n_threads, _n_sub
        float _thr
        double acc
        float [:, :, ::1] streamlines
        float [:, :, ::1] streamlines_bundle
        double [:, ::1] streamline_sub
        double [:, ::1] dct_M
        float [::1] lengths = np.empty(3000, dtype=np.float32)

    t0 = time()
    set_verbose('tractogram', verbose)
    logger.info('Searching for similar streamlines')

    files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck'])]
    files = [File(name=f'bundle_filenames_{i}', type_='input', path=f, ext='.tck') for i, f in enumerate(bundle_filenames)]
    dirs  = [Dir(name='out_folder', path=out_folder)]
    nums  = [
        Num(name='thr', value=thr, min_=0.1),
        Num(name='n_sub', value=n_sub, min_=2),
        Num(name='n_dct', value=n_dct, min_=2, max_=n_sub),
        Num(name='n_threads', value=n_threads, min_=0),
    ]
    check_params(files=files, dirs=dirs, nums=nums, force=force)
    _n_threads = n_threads if n_threads is not None else 0
    _n_sub = n_sub
    _thr = thr

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    try:
        streamline_sub = np.empty((n_sub,3), dtype=np.float64)
        dct_M = dct( np.eye(n_sub), axis=0, norm="ortho" )[:n_dct,:]

        # load and convert streamlines of tractogram
        tt = time()
        logger.subinfo(f"Loading tractogram:", indent_char='*', indent_lvl=1, with_progress=True)
        TCK_in = LazyTractogram(tractogram_filename, mode='r')
        n_streamlines = int(TCK_in.header['count'])
        streamlines = np.empty((n_streamlines, n_dct, 3), dtype=np.float32)
        with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
            for i in range(n_streamlines):
                TCK_in.read_streamline()
                set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline_sub, n_sub, lengths)
                for j in range(n_dct):
                    for k in range(3):
                        acc = 0.0
                        for l in range(n_sub):
                            acc = acc + dct_M[j, l] * streamline_sub[l, k]
                        streamlines[i,j,k] = acc
                pbar.update()
        TCK_in.close()
        logger.subinfo(f'{n_streamlines} streamlines loaded', indent_lvl=2, indent_char='-')
        logger.debug( f'Tractogram load/resample time = {time()-tt:.3f}s' )

        # load and convert streamlines of each bundle
        logger.subinfo(f'Searching streamlines in {len(bundle_filenames)} bundle(s):', indent_char='*', indent_lvl=1, with_progress=True)
        with ProgressBar(total=len(bundle_filenames), disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
            for bundle_filename in bundle_filenames:
                tt = time()
                logger.debug(f'Loading "{bundle_filename}"')
                TCK_in = LazyTractogram(bundle_filename, mode='r')
                n_streamlines_bundle = int(TCK_in.header['count'])
                streamlines_bundle = np.empty((n_streamlines_bundle, n_dct, 3), dtype=np.float32)
                for i in range(n_streamlines_bundle):
                    TCK_in.read_streamline()
                    set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline_sub, n_sub, lengths)
                    for j in range(n_dct):
                        for k in range(3):
                            acc = 0.0
                            for l in range(n_sub):
                                acc = acc + dct_M[j, l] * streamline_sub[l, k]
                            streamlines_bundle[i,j,k] = acc
                TCK_in.close()
                logger.debug(f'  - {n_streamlines_bundle} streamlines loaded')
                logger.debug(f'  - Load/resample time = {time()-tt:.3f}s')

                # streamline search
                logger.debug(f"Searching")
                is_found = np.empty(n_streamlines, dtype=np.uint8)
                tt = time()
                for i in prange(n_streamlines, nogil=True, schedule='static', chunksize=10000, num_threads=_n_threads):
                    is_found[i] = _is_within_distance(streamlines, i, streamlines_bundle, _thr, _n_sub)
                logger.debug(f'  - {np.count_nonzero(is_found)} streamlines recognized')
                logger.debug(f'  - Bundle search time = {time()-tt:.3f}s')

                # saving output tractogram
                logger.debug(f"Saving recognized bundle")
                TCK_in = LazyTractogram(tractogram_filename, mode='r')
                out_tractogram_filename = os.path.splitext(os.path.basename(bundle_filename))[0]
                out_tractogram_filename = os.path.join(out_folder, out_tractogram_filename+f'__thr={thr:.1f}'+suffix+'.tck')
                TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
                count = 0
                for i in range(n_streamlines):
                    TCK_in.read_streamline()
                    if is_found[i]:
                        TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                        count += 1
                TCK_out.close(write_eof=True, count=count)
                TCK_in.close()

                pbar.update()

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close()
        logger.info( f'[ {format_time(time() - t0)} ]' )
        return is_found
