# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
cimport cython
from libc.math cimport isinf, isnan, NAN
from libc.stdio cimport fclose, fgets, fopen, fread, fseek, fwrite, SEEK_CUR, SEEK_END, SEEK_SET
from libc.string cimport strchr, strlen, strncmp
from libcpp.string cimport string

from dicelib.tractogram cimport LazyTractogram
from dicelib.streamline import is_flipped
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, Dir, File, Num, format_time

import os
import numpy as np
from time import time

cdef float[1] NAN1 = {NAN}
cdef float[3] NAN3 = {NAN, NAN, NAN}

logger = setup_logger('tsf')


@cython.final
cdef class TrackScalarFile:
    """Class to read/write scalars along streamlines from Track Scalar Files (TSF) one by one.

    A file can be opened in three different modalities:
    - 'r': reading
    - 'w': writing
    - 'a': appending

    TODO: complete this description.
    """
    # cdef readonly   str                             filename
    # cdef readonly   str                             suffix
    # cdef readonly   dict                            header
    # cdef readonly   str                             mode
    # cdef readonly   bint                            is_open
    # cdef readonly   float[:]                        scalars
    # cdef readonly   unsigned int                    n_pts
    # cdef            int                             max_points
    # cdef            FILE*                           fp

    def __init__( self, char *filename, char* mode, header=None, unsigned int max_points=3000 ):
        """Initialize the class.

        Parameters
        ----------
        filename : str
            Name of the TSF file.
        mode : str
            Opens the file for reading ('r'), writing ('w') or appending ('a') scalar values.
        header : dictionary, optional
            A dictionary of 'key: value' pairs that define the items in the header.
        max_points : unsigned int, default=3000
            The maximum number of scalars allowed for a streamline.
        """
        self.is_open = False
        self.filename = filename
        _, self.suffix = os.path.splitext( filename )
        if self.suffix not in ['.tsf']:
            raise ValueError( 'Only ".tsf" files are supported for now.' )

        if mode not in ['r', 'w', 'a']:
            raise ValueError( '"mode" must be either "r", "w" or "a"' )
        self.mode = mode

        if mode=='r':
            if max_points<=0:
                raise ValueError( '"max_points" should be positive' )
            self.max_points = max_points
            self.scalars = np.empty( (max_points,), dtype=np.float32 )
        else:
            self.scalars = None
        self.n_pts = 0

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
            self._write_header( header )
        else:
            # file is open for appending => move pointer to end
            fseek( self.fp, 0, SEEK_END )

        self.is_open = True


    cpdef int read_scalars( self ) nogil:
        """Read next streamline's values from the current position in the file.

        Returns
        -------
        output : int
            Number of points/coordinates read from disk.
        """
        cdef float scalar
        cdef int n_read
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )

        self.n_pts = 0
        while True:
            if self.n_pts>self.max_points:
                raise RuntimeError( f'Problem reading data, scalars seem too many (>{self.max_points} points)' )

            n_read = fread( &self.scalars[0], 4, 1, self.fp )
            if n_read < 1:
                return 0
            if isnan(self.scalars[0]):
                break
            if isinf(self.scalars[0]):
                break
            self.n_pts += 1

        return self.n_pts


    cpdef void write_scalars( self, float [:] scalars, int n=-1 ) nogil:
        """Write streamline's scalars at the current position in the file.

        Parameters
        ----------
        scalars : Nx3 numpy array
            The streamline's scalars to write
        n : int
            Writes first n values of the streamline. If n<0 (default), writes all scalars.
            NB: be careful because, for efficiency, scalars are represented as a fixed-size array
        """
        # cdef float [::1] scalars_arr = scalars
        # cdef int [::1]  pts_arr = pts
        # cdef int        i = 0
        # cdef int        sum_len = 0
        if n<0:
            n = scalars.shape[0]
        if n==0:
            return

        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode=='r':
            raise RuntimeError( 'File is not open for writing/appending' )

        # write scalars data
        # for i in range(pts_arr.size):
        #     if fwrite( &scalars_arr[sum_len],4, pts_arr[i], self.fp )!=pts_arr[i]:
        #         raise IOError( 'Problems writing scalars data to file' )
        #     sum_len += pts_arr[i]
        if fwrite( &scalars[0], 4, n, self.fp )!=n:
            raise IOError( 'Problems writing scalars data to file' )
        # write end-of-scalars signature
        fwrite( NAN1, 4, 1, self.fp )


    cpdef close( self, bint write_eof=True, int count=-1 ):
        """Close the file associated with this Track Scalar File.

        Parameters
        ----------
        write_eof : bool, default=True
            Write the EOF marker, i.e. INF, at the current position.
            NB: use at your own risk if you know what you are doing.
        count : int, default=-1
            Update the 'count' field in the header with this value (if -1, then do not update)
        """
        cdef float inf = float('inf')

        if self.is_open==False:
            return

        if self.mode!='r':
            # write end-of-file marker
            if write_eof:
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
        the binary data part of the file, ready to read scalars.
        """
        cdef char[5000000] line # a field can be max 5MB long
        cdef int           nLines = 0

        if len(self.header) > 0:
            raise RuntimeError( 'Header already read' )

        # check if it's a valid TSF file
        fseek( self.fp, 0, SEEK_SET )
        if fgets( line, sizeof(line), self.fp )==NULL:
            raise IOError( 'Problems reading header from file FIRST LINE' )
        if line.strip() != 'mrtrix tracks scalars':
            raise IOError( f'"{self.filename}" is not a valid TSF file' )

        # parse one line at a time
        while True:
            if nLines>=1000:
                raise RuntimeError( 'Problem parsing the header; too many header lines' )
            if fgets( line, sizeof(line), self.fp )==NULL:
                raise IOError( 'Problems reading header from file' )
            line[strlen(line)-1] = 0
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

        # check if the 'count' field is present TODO: fix this, allow working even without it
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
        the binary data part of the file, ready to write scalars.

        Parameters
        ----------
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header.
        """
        cdef string line
        cdef int offset = 25 # accounts for 'mrtrix track scalars\n' and 'END\n'

        if header is None or type(header)!=dict:
            raise RuntimeError( 'Provided header is empty or invalid' )

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if 'count' not in header:
            raise RuntimeError( 'Problem parsing the header; field "count" not found' )
        if type(header['count'])==list:
            raise RuntimeError( 'Problem parsing the header; field "count" has multiple values' )

        fseek( self.fp, 0, SEEK_SET )
        line = b'mrtrix track scalars\n'
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

        if "timestamp" not in header:
            line = f'timestamp: {time()}\n'
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


    def __dealloc__( self ):
        if self.is_open:
            fclose( self.fp )


# def join( input_tsf: List[str], output_tsf: str, verbose: int=3, force: bool=False ):
#     """Join multiple tsf files into a single tsf file.

#     Parameters
#     ----------
#     input_tsf: list
#         List of paths to the input tsf files.
#     output_tsf: str
#         Path to the output tsf file.
#     """
#     set_verbose('tractogram', verbose)

#     files = [File(name='output_tsf', type_='output', path=output_tsf, ext='.tsf')]
#     for i, tsf in enumerate(input_tsf):
#         files.append(File(name=f'input_tsf_{i}', type_='input', path=tsf, ext='.tsf'))
#     check_params(files=files, force=force)

#     header = TrackScalarFile(input_tsf[0], 'r').header
#     Tsf_out = TrackScalarFile(output_tsf, 'w', header=header)

#     final_pts = 0
#     for tsf in input_tsf:
#         Tsf_in = TrackScalarFile(tsf, 'r')
#         scalar_list, n_pts_list = Tsf_in.read_scalar()
#         final_pts += int(Tsf_in.header['count'])
#         Tsf_out.write_scalar(scalar_list, n_pts_list)
#         Tsf_in.close()
#     # update the count in the header ensuring the same number of characters
#     Tsf_out.close(write_eof=True, count=final_pts)



#NOTE: check if these functions are needed

# def _color_by_scalar_file(TCK_in, values, num_streamlines):
#     """Color streamlines based on sections.

#     Parameters
#     ----------
#     TCK_in: array
#         Input LazyTractogram object.
#     values: list
#         List of scalars used to color the streamlines.

#     Returns
#     -------
#     array
#         Array mapping scalar values to each vertex of each streamline.
#     array
#         Array containing the number of points of each input streamline.
#     """
#     scalar_list = []
#     n_pts_list = []
#     for i in range(num_streamlines):
#         TCK_in.read_streamline()
#         n_pts_list.append(TCK_in.n_pts)
#         streamline_points = np.arange(TCK_in.n_pts)
#         resample = np.linspace(0, TCK_in.n_pts, len(values), endpoint=True, dtype=np.int32)
#         streamline_points = np.interp(streamline_points, resample, values)
#         scalar_list.extend(streamline_points)
#     return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)

# cpdef create( tractogram_filename: str, scalars_filename: str, out_tsf_filename: str, check_orientation: bool=False, out_tractogram_filename: str=None, force: bool=False, verbose: int=3 ):
#     """Create a TSF file for each streamline in order to color them for visualization.

#     Parameters
#     ----------
#     tractogram_filename : str
#         Path to the tractogram (.tck) containing the streamlines to process.
#     scalars_filename : str
#         Path to the file (.txt, .npy) containing the scalars at each streamline's coordinate to be used
#         for coloring the streamlines along their trajectories.
#     out_tsf_filename : str
#         Path to the output tsf file (.???).
#     check_orientation : bool, default=False
#         If True, create a new tractogram with the streamlines oriented in the same direction.
#     out_tractogram_filename : string, optional
#         !!! MISSING DOCUMENTATION !!!
#     """
#     set_verbose('tractogram', verbose)

#     if check_orientation:
#         if out_tractogram_filename is None:
#             raise ValueError("Please specify an output tractogram")

#     files = [File(name='tractogram_filename', type_='input', path=tractogram_filename, ext='.tck'),
#             File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']),
#             File(name='out_tsf_filename', type_='output', path=out_tsf_filename, ext='.tsf')]
#     if out_tractogram_filename:
#         files.append( File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext='.tck') )

#     if check_orientation:
#         check_params(files=files, force=force)
#     elif scalars_filename:
#         files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']))
#         check_params(files=files, force=force)
#     else:
#         raise ValueError("Please specify a color option")

#     cdef float[:,::1] ref_streamline = np.empty((2000,3), dtype=np.float32)
#     cdef float[:,::1] streamline_out = np.empty((2000,3), dtype=np.float32)
#     if check_orientation:
#         TCK_in = LazyTractogram(tractogram_filename, mode='r')
#         num_streamlines = int(TCK_in.header['count'])
#         TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
#         TCK_in.read_streamline()
#         ref_streamline[:TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts].copy()
#         ref_n_pts = TCK_in.n_pts
#         with ProgressBar( total=num_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
#             for i in range(int(num_streamlines)-1):
#                 TCK_in.read_streamline()
#                 flip = is_flipped(TCK_in.streamline[:TCK_in.n_pts], ref_streamline[:ref_n_pts])
#                 if flip:
#                     streamline_out[:TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts][::-1]
#                 else:
#                     streamline_out[:TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts]
#                 TCK_out.write_streamline(streamline_out, TCK_in.n_pts)
#                 pbar.update()
#         TCK_out.close()
#         TCK_in.close()
#         TCK_in = LazyTractogram(out_tractogram_filename, mode='r')
#         num_streamlines = TCK_in.header['count']
#     else:
#         TCK_in = LazyTractogram(tractogram_filename, mode='r')
#         num_streamlines = TCK_in.header['count']

#     if scalars_filename.endswith('.txt'):
#         values = np.loadtxt(scalars_filename)
#     else:
#         values = np.load(scalars_filename)
#     scalar_arr, n_pts_list = _color_by_scalar_file(TCK_in, values, int(num_streamlines))

#     tsf = TrackScalarFile(out_tsf_filename, 'w', header=TCK_in.header)
#     tsf.write_scalar(scalar_arr, n_pts_list)


# def create_color_scalar_file(streamline, num_streamlines):
#     """Create a scalar file for each streamline in order to color them.
#
#     Parameters
#     ----------
#     streamlines: list
#         List of streamlines.
#
#     Returns
#     -------
#     str
#         Path to scalar file.
#     """
#     scalar_list = list()
#     n_pts_list = list()
#     for i in range(num_streamlines):
#         # pt_list = list()
#         streamline.read_streamline()
#         n_pts_list.append(streamline.n_pts)
#         for j in range(streamline.n_pts):
#             scalar_list.extend([float(j)])
#         # scalar_list.append(pt_list)
#     return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)