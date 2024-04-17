# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False

from dicelib.streamline import apply_smoothing, length as streamline_length, rdp_reduction, resample as s_resample, set_number_of_points, smooth
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, Dir, File, Num, format_time

import ast
import os
import random as rnd
import sys
import shutil

import nibabel as nib
import numpy as np

from dicelib.streamline cimport apply_affine_1pt

from libc.math cimport isinf, isnan, NAN, sqrt
from libc.stdio cimport fclose, fgets, fopen, fread, fseek, fwrite, SEEK_CUR, SEEK_END, SEEK_SET
from libc.stdlib cimport malloc, free
from libcpp cimport bool as cbool
from libc.string cimport strchr, strlen, strncmp 
from libcpp.string cimport string

from time import time


cdef float[1] NAN1 = {NAN}
cdef float[3] NAN3 = {NAN, NAN, NAN}

logger = setup_logger('tractogram')

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
        filename : string
            Name of the file containing the tractogram to open.
        mode : string
            Opens the tractogram for reading ('r'), writing ('w') or appending ('a') streamlines.
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header; this parameter is only required
            when writing streamlines to disk (default : None).
        max_points : unsigned int
            The maximum number of points/coordinates allowed for a streamline (default : 3000).
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
            self._write_header( header )
        else:
            # file is open for appending => move pointer to end
            fseek( self.fp, 0, SEEK_END )

        self.is_open = True


    cdef int _read_streamline( self ) nogil:
        """Read next streamline from the current position in the file.

        For efficiency reasons, multiple streamlines are simultaneously loaded from disk using a buffer.
        The current streamline is stored in the fixed-size numpy array 'self.streamline' and its actual
        length, i.e., number of points/coordinates, is stored in 'self.n_pts'.

        Returns
        -------
        output : int
            Number of points/coordinates read from disk.
        """
        cdef float* ptr = &self.streamline[0,0]
        cdef int    n_read
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )

        self.n_pts = 0
        while True:
            if self.n_pts>self.max_points:
                raise RuntimeError( f'Problem reading data, streamline seems too long (>{self.max_points} points)' )
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

    cpdef int read_streamline( self ):
        """Read next streamline from the current position in the file.

        For efficiency reasons, multiple streamlines are simultaneously loaded from disk using a buffer.
        The current streamline is stored in the fixed-size numpy array 'self.streamline' and its actual
        length, i.e., number of points/coordinates, is stored in 'self.n_pts'.

        Returns
        -------
        output : int
            Number of points/coordinates read from disk.
        """
        cdef float* ptr = &self.streamline[0,0]
        cdef int    n_read
        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode!='r':
            raise RuntimeError( 'File is not open for reading' )

        self.n_pts = 0
        while True:
            if self.n_pts>self.max_points:
                raise RuntimeError( f'Problem reading data, streamline seems too long (>{self.max_points} points)' )
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


    cdef void _write_streamline( self, float [:,:] streamline, int n=-1 ) nogil:
        """Write a streamline at the current position in the file.

        Parameters
        ----------
        streamline : Nx3 numpy array
            The streamline data
        n : int
            Writes first n points of the streamline. If n<0 (default), writes all points.
            NB: be careful because, for efficiency, a streamline is represented as a fixed-size array
        """
        if streamline.shape[0]>1 or streamline.shape[1]!=3:
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

    cpdef write_streamline( self, float [:,:] streamline, int n=-1 ):
        """Write a streamline at the current position in the file.

        Parameters
        ----------
        streamline : Nx3 numpy array
            The streamline data
        n : int
            Writes first n points of the streamline. If n<0 (default), writes all points.
            NB: be careful because, for efficiency, a streamline is represented as a fixed-size array
        """
        if  streamline.ndim!=2 or streamline.shape[1]!=3:
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
        cdef size_t        max_size_line = 5000000*sizeof(char) # 5MB
        cdef char*         line = <char*> malloc(max_size_line)
        cdef int           nLines = 0

        if len(self.header) > 0:
            raise RuntimeError( 'Header already read' )

        fseek( self.fp, 0, SEEK_SET )

        # check if it's a valid TCK file
        if fgets(line, max_size_line, self.fp) == NULL:
            raise IOError( 'Problems reading header from file FIRST LINE' )
        if line.strip() != 'mrtrix tracks':
            raise IOError( f'"{self.filename}" is not a valid TCK file' )

        # parse one line at a time
        while True:
            if nLines>=1000:
                raise RuntimeError( 'Problem parsing the header; too many header lines' )
            if fgets(line, max_size_line, self.fp) == NULL:
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


cdef class Tsf:
    """Class to read/write tsf files for visualization.

    A file can be opened in three different modalities:
    - 'r': reading
    - 'w': writing
    - 'a': appending

    TODO: complete this description.
    """
    cdef readonly   str                             filename
    cdef readonly   str                             suffix
    cdef readonly   dict                            header
    cdef readonly   str                             mode
    cdef readonly   bint                            is_open
    cdef readonly   unsigned int                    n_pts
    cdef            FILE*                           fp

    def __init__( self, char *filename, char* mode, header=None ):
        """Initialize the class.

        Parameters
        ----------
        filename : string
            Name of the tsf file.
        mode : string
            Opens the file for reading ('r'), writing ('w') or appending ('a') scalar values.
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header;

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
            # TODO
            pass
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

    cpdef _read_header( self ):
        """Read the header from file.
        After the reading, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to read scalars.
        """
        cdef char[5000000] line # a field can be max 5MB long
        cdef char*         ptr
        cdef int           nLines = 0

        if len(self.header) > 0:
            raise RuntimeError( 'Header already read' )

        fseek( self.fp, 0, SEEK_SET )

        # check if it's a valid tsf file
        if fgets( line, sizeof(line), self.fp )==NULL:
            raise IOError( 'Problems reading header from file FIRST LINE' )
        # line[strlen(line)-1] = 0
        if strncmp( line, 'mrtrix track scalars', 20)!=0:
            raise IOError( f'"{self.filename}" is not a valid tsf file' )

        # parse one line at a time
        while True:
            if nLines>=1000:
                raise RuntimeError( 'Problem parsing the header; too many header lines' )
            if fgets( line, sizeof(line), self.fp )==NULL:
                raise IOError( 'Problems reading header from file' )
            line[strlen(line)-1] = 0
            if strncmp(line,'END',3)==0:
                break
            ptr = strchr(line, ord(':'))
            if ptr==NULL:
                raise RuntimeError( 'Problem parsing the header; format not valid' )
            key = str(line[:(ptr-line)])
            val = ptr+2
            if key not in self.header:
                self.header[key] = val
            else:
                if type(self.header[key])!=list:
                    self.header[key] = [ self.header[key] ]
                self.header[key].append( val )
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
        cdef int offset = 25 # accounts for 'mrtrix tracks\n' and 'END\n'

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
            line = f'timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
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

    cpdef write_scalar( self, scalars, pts):
        """Write scalars at the current position in the file.

        Parameters
        ----------
        scalars : numpy array
            The scalar values to write.
        """
        cdef float [::1] scalars_arr = scalars
        cdef int [::1]  pts_arr = pts
        cdef int        i = 0
        cdef int        sum_len = 0
        if  scalars.ndim!=1:
            raise RuntimeError( 'array must be one dimension' )

        if self.is_open==False:
            raise RuntimeError( 'File is not open' )
        if self.mode=='r':
            raise RuntimeError( 'File is not open for writing/appending' )

        # write scalars data
        for i in range(pts_arr.size):
            if fwrite( &scalars_arr[sum_len],4, pts_arr[i], self.fp )!=pts_arr[i]:
                raise IOError( 'Problems writing scalars data to file' )
            sum_len += pts_arr[i]
            # write end-of-scalars signature
            fwrite( NAN1, 4, 1, self.fp )


    cpdef close( self, bint write_eof=True, int count=-1 ):
        """Close the file associated with the tractogram.

        Parameters
        ----------
        write_eof : bool
            Write the EOF marker, i.e. INF, at the current position (default : True).
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


    def __dealloc__( self ):
        if self.is_open:
            fclose( self.fp )


def get_indices_of_streamlines( needle_filename: str, haystack_filename: str, idx_out: str=None, verbose: int=3, force: bool=False ) -> np.ndarray:
    """Finds the indices of the streamlines in a subset of streamlines from a larger tractogram.

    Parameters
    ----------
    needle_filename : string
        Path to the file (.tck) containing the subset of streamlines to find.

    haystack_filename : string
        Path to the file (.tck) containing the full set of streamlines in which to search.

    Returns
    -------
    idx : numpy array
        Indices of the streamlines from 'needle' that where found in 'haystack'.
    """

    set_verbose('tractogram', verbose)

    logger.info('Finding indices of streamlines')

    files = [File(name='needle_filename', type_='input', path=needle_filename, ext='.tck'),
             File(name='haystack_filename', type_='input', path=haystack_filename, ext='.tck')]
    if idx_out:
        files.append(File(name='idx_out', type_='output', path=idx_out, ext=['.txt', '.npy']))
        
    check_params(files=files, force=force)

    TCK_haystack = LazyTractogram( haystack_filename, mode='r' )
    n_haystack = int( TCK_haystack.header['count'] )
    TCK_needle = LazyTractogram( needle_filename, mode='r' )
    n_needle = int( TCK_needle.header['count'] )
    t0 = time()

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
        

    indices = np.flatnonzero( np.in1d( hash_all, hash_subset, assume_unique=True ) )
    logger.subinfo(f'Number of streamlines found: {len(indices)}', indent_lvl=1, indent_char='*')
    # save the indices to file
    if idx_out:
        # check if .txt or .npy
        if idx_out.endswith('.txt'):
            np.savetxt( idx_out, indices, fmt='%d' )
        elif idx_out.endswith('.npy'):
            np.save( idx_out, indices )
    # return indices of the streamlines that were found
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
    return indices


def create_color_scalar_file(streamline, num_streamlines):
        """
        Create a scalar file for each streamline in order to color them.
        Parameters
        ----------
        streamlines: list
            List of streamlines.
        Returns
        -------
        scalar_file: str
            Path to scalar file.
        """
        scalar_list = list()
        n_pts_list = list()
        for i in range(num_streamlines):
            # pt_list = list()
            streamline.read_streamline()
            n_pts_list.append(streamline.n_pts)
            for j in range(streamline.n_pts):
                scalar_list.extend([float(j)])
            # scalar_list.append(pt_list)
        return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)


def color_by_scalar_file(TCK_in, values, num_streamlines):
    """
    Color streamlines based on sections.
    Parameters
    ----------
    TCK_in: array
        Input LazyTractogram object.
    values: list
        List of scalars used to color the streamlines.
    Returns
    -------
    array
        Array mapping scalar values to each vertex of each streamline.
    array
        Array containing the number of points of each input streamline.
    """
    scalar_list = []
    n_pts_list = []
    for i in range(num_streamlines):
        streamline = TCK_in.read_streamline()
        n_pts_list.append(streamline.n_pts)
        streamline_points = np.arange(streamline.n_pts)
        resample = np.linspace(0, streamline.n_pts, len(values), endpoint=True, dtype=np.int32)
        streamline_points = np.interp(streamline_points, resample, values)
        scalar_list.extend(streamline_points)
    return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)


def tsf_create( input_tractogram: str, output_tsf: str, orientation: bool=False, file: str=None, verbose: int=3, force: bool=False ):
    """Create a tsf file for each streamline in order to color them for visualization.
    
    Parameters
    ----------
    input_tractogram: str
        Path to the input tractogram.
    output_tsf: str
        Path to the output tsf file.
    orientation: bool
        If True, color the streamlines based on their orientation.
    file: str
        Path to the file containing the scalar values used to color the streamlines.
    """

    set_verbose('tractogram', verbose)
    
    files = [File(name='input_tractogram', type_='input', path=input_tractogram, ext='.tck'),
             File(name='output_tsf', type_='output', path=output_tsf, ext='.tsf')]

    if orientation:
        check_params(files=files, force=force)
    elif file:
        files.append(File(name='file', type_='input', path=file))
        check_params(files=files, force=force)
    else:
        raise ValueError("Please specify a color option")

    TCK_in = LazyTractogram(input_tractogram, mode='r')
    num_streamlines = TCK_in.header['count']

    if orientation:
        scalar_arr, n_pts_list = create_color_scalar_file(TCK_in, int(num_streamlines))
    else:
        values = np.loadtxt(file)
        scalar_arr, n_pts_list = color_by_scalar_file(TCK_in, values, int(num_streamlines))

    tsf = Tsf(output_tsf, 'w', header=TCK_in.header)
    tsf.write_scalar(scalar_arr, n_pts_list)


def compute_lengths( input_tractogram: str, output_scalar_file: str=None, verbose: int=3, force: bool=False ) -> np.ndarray:
    """Compute the lengths of the streamlines in a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    Returns
    -------
    lengths : array of double
        Lengths of all streamlines in the tractogram [in mm]
    """

    set_verbose('tractogram', verbose)

    files = [File(name='input_tractogram', type_='input', path=input_tractogram)]
    if output_scalar_file is not None:
        files.append(File(name='output_scalar_file', type_='output', path=output_scalar_file, ext=['.txt', '.npy']))
    check_params(files=files, force=force)

    #----- iterate over input streamlines -----
    TCK_in = None
    lengths = None
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        if n_streamlines <= 0:
            logger.error('The tractogram is empty')

        logger.info('Streamline lengths')
        t0 = time()
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
            logger.subinfo(f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*', indent_lvl=1)
            logger.subinfo(f'min: {lengths.min():.3f}  max: {lengths.max():.3f}  mean: {lengths.mean():.3f}  std: {lengths.std():.3f}', indent_char='*', indent_lvl=1)

        if output_scalar_file is None:
            return streamline_length
        else:
            output_scalar_file_ext = os.path.splitext(output_scalar_file)[1]
            if output_scalar_file_ext == '.txt':
                np.savetxt(output_scalar_file, lengths, fmt='%.4f')
            elif output_scalar_file_ext == '.npy':
                np.save(output_scalar_file, lengths, allow_pickle=False)

    except Exception as e:
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
    


def info( input_tractogram: str, compute_lengths: bool=False, max_field_length: int=None, verbose: int=3 ):
    """Print some information about a tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    compute_lengths : boolean
        Show stats on streamline lengths (default : False).

    max_field_length : int
        Maximum length allowed for printing a field value (default : all chars)

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).
    """
    set_verbose('tractogram', verbose)

    files = [File(name='input_tractogram', type_='input', path=input_tractogram, ext='.tck')]
    nums = None
    if max_field_length is not None:
        nums = [Num(name='max_field_length', value=max_field_length, min_=25)]
    check_params(files=files, nums=nums)

    #----- iterate over input streamlines -----
    TCK_in  = None
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

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


def filter( input_tractogram: str, output_tractogram: str, minlength: float=None, maxlength: float=None, minweight: float=None, maxweight: float=None, weights_in: str=None, weights_out: str=None, random: float=1.0, verbose: int=3, force: bool=False ):
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
        Randomly keep the given percentage of streamlines: 0=discard all, 1=keep all. 
        This filter is applied after all others (default : 1).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """
    set_verbose('tractogram', verbose)

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='output_tractogram', type_='output', path=output_tractogram, ext='.tck')
    ]
    if weights_in is not None:
        files.append(File(name='weights_in', type_='input', path=weights_in, ext=['.txt', '.npy']))
        weights_in_ext = os.path.splitext(weights_in)[1]
    if weights_out is not None:
        weights_out_ext = os.path.splitext(weights_out)[1]
        files.append(File(name='weights_out', type_='output', path=weights_out, ext=['.txt', '.npy']))
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
    if random != 1:
        messages.append(f'Randomly keeping {random * 100:.0f}% of the streamlines')
    check_params(files=files, nums=nums, force=force)

    logger.info('Filtering tractogram')
    t0 = time()
    for msg in messages:
        logger.subinfo(msg, indent_char='*', indent_lvl=1)

    if weights_in is not None:
        if weights_in_ext == '.txt':
            w = np.loadtxt(weights_in).astype(np.float64)
        elif weights_in_ext == '.npy':
            w = np.load(weights_in, allow_pickle=False).astype(np.float64)
        logger.subinfo('Using streamline weights from text file', indent_char='*', indent_lvl=1)
    else:
        w = np.array([])

    n_written = 0
    TCK_in  = None
    TCK_out = None

    #----- iterate over input streamlines -----
    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        n_streamlines = int( TCK_in.header['count'] )
        # open the outut file
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_char='*', indent_lvl=1)
        if weights_in is not None and n_streamlines!=w.size:
            logger.error(f'Number of weights is different from number of streamlines ({w.size},{n_streamlines})')

        with ProgressBar( total=2*n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            # check if #(weights)==n_streamlines
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
                if weights_in is not None and (
                    (minweight is not None and w[i]<minweight) or
                    (maxweight is not None and w[i]>maxweight)
                ):
                    kept[i] = False
                    continue

                pbar.update()

            TCK_in._seek_origin(int(TCK_in.header['file'][2:]))

            if random < 1:
                idx_true = np.where(kept == True)[0]
                discard_choice = np.random.choice( idx_true, int(idx_true.size * (1-random)), replace=False )
                kept[discard_choice] = False
                
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if kept[i]:
                    TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                    n_written += 1
                pbar.update()
            
            if weights_out is not None and w.size > 0:
                if weights_out_ext == '.txt':
                    np.savetxt(weights_out, w[kept == True].astype(np.float32), fmt='%.5e')
                elif weights_out_ext == '.npy':
                    np.save(weights_out, w[kept == True].astype(np.float32), allow_pickle=False)


    except Exception as e:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( output_tractogram ):
            os.remove( output_tractogram )
        if weights_out is not None and os.path.isfile( weights_out ):
            os.remove( weights_out )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        logger.subinfo(f'Number of streamlines written: {n_written}', indent_char='*', indent_lvl=1)
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close(write_eof=True, count=n_written )
    

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


def split( input_tractogram: str, input_assignments: str, output_folder: str='bundles', regions_in: str=None, weights_in: str=None, max_open: int=None, prefix: str='bundle_', verbose: int=3, force: bool=False, log_list=None ):
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

    regions_in : list of integers
        If a single integer is provided, only streamlines assigned to that region will be extracted.
        If two integers are provided, only streamlines connecting those two regions will be extracted.

    weights_in : string
        Text file with the input streamline weights (one row/streamline). If not None, one individual
        file will be created for each splitted tractogram, using the same filename prefix.

    max_open : integer
        Maximum number of files opened at the same time (default : None).
        If the specified value exceeds the system limit, an attempt is made to increase the latter so that `max_open` equals the 90% of the system limit
        If None, the following values are used:
            - on Unix: 90% of half the default system hard limit
            - on Windows: 90% of twice the default system limit

    prefix : string
        Prefix for the output filenames (default : 'bundle_').

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('tractogram', verbose)

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='input_assignments', type_='input', path=input_assignments)
    ]
    if weights_in is not None:
        files.append(File(name='weights_in', type_='input', path=weights_in, ext=['.txt', '.npy']))
        weights_in_ext = os.path.splitext(weights_in)[1]
    dirs = [Dir(name='output_folder', path=output_folder)]
    check_params(files=files, dirs=dirs, force=force)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def split_regions(input_string):
        try:
            # ast.literal_eval safely parses an input string to a Python literal structure
            return ast.literal_eval(input_string)
        except (SyntaxError, ValueError):
            # Handle the exception if the input string is not a valid Python literal structure
            logger.error('The input string is not a valid Python literal structure.')
            return None

    if not regions_in==None:
        if not isinstance(split_regions(regions_in), (list, tuple, int)):
            logger.error('Invalid regions input')
        else:
            regions_str = "[]," + regions_in
            regions = []
            for r in split_regions(regions_str):
                if r == []:
                    continue
                if isinstance(r, list):
                    if len(r) != 2:
                        logger.error('Invalid regions input')
                regions.append(r)
    else:
        regions = []

    if weights_in is not None:
        if weights_in_ext == '.txt':
            w = np.loadtxt(weights_in).astype(np.float64)
        elif weights_in_ext == '.npy':
            w = np.load(weights_in, allow_pickle=False).astype(np.float64)
        w_idx = np.zeros_like(w, dtype=np.int32)

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
    
    logger.info(f'Splitting tractogram')
    t0 = time()
    try:
        logger.subinfo(f'Number of input streamline weights: {w.size}', indent_char='*', indent_lvl=1)
    except UnboundLocalError:
        pass
    logger.subinfo(f'Output tractograms path: \'{output_folder}\'', indent_char='*', indent_lvl=1)
    logger.debug(f'Number of files opened simultaneously: {max_open}')

    #----- iterate over input streamlines -----
    TCK_in          = None
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
        logger.subinfo(f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*', indent_lvl=1)

        # open the assignments
        if os.path.splitext(input_assignments)[1]=='.txt':
            assignments = np.loadtxt( input_assignments, dtype=np.int32 )
        elif os.path.splitext(input_assignments)[1]=='.npy':
            assignments = np.load( input_assignments, allow_pickle=False ).astype(np.int32)
        else:
            logger.error('Invalid extension for the assignments file')
        if assignments.ndim!=2 or assignments.shape[1]!=2:
            print( (assignments.ndim, assignments.shape))
            logger.error('Unable to open assignments file')
        logger.subinfo(f'Number of assignments in input file: {assignments.shape[0]}', indent_char='*', indent_lvl=1)

        # check if #(assignments)==n_streamlines
        if n_streamlines!=assignments.shape[0]:
            logger.error(f'Number of assignments is different from number of streamlines ({assignments.shape[0]},{n_streamlines})')
        # check if #(weights)==n_streamlines
        if weights_in is not None and n_streamlines!=w.size:
            logger.error(f'# of weights ({w.size}) is different from # of streamlines ({n_streamlines})')

        # create empty tractograms for unique assignments
        if len(regions)==0:
            unique_assignments = np.unique(assignments, axis=0)
        else:
            unique_assignments = []
            assignments.sort()
            for r in regions:
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
            tmp = LazyTractogram( os.path.join(output_folder,f'{pref_key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if weights_in is not None:
                WEIGHTS_out_idx[key] = i+1

        # add key for non-connecting streamlines
        if unassigned_count and len(regions)==0:
            key = 'unassigned'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTractogram( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
            if weights_in is not None:
                WEIGHTS_out_idx[key] = 0

        logger.debug(f'Created {len(TCK_outs)} empty files for output tractograms')

        #----  iterate over input streamlines  -----
        n_file_open = 0
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                # skip assignments not in the regions
                if len(regions) > 0:
                    skip = True
                    for r in regions:
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
                    fname = os.path.join(output_folder,f'{pref_key}.tck')
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
            logger.subinfo(f'Saving one weights file per bundle', indent_char='*', indent_lvl=1)
            with ProgressBar(disable=verbose < 3, hide_on_exit=True) as pbar:
                for key in WEIGHTS_out_idx.keys():
                    if key == 'unassigned':
                        pref_key = 'unassigned'
                    else:
                        pref_key = f'{prefix}{key}'
                    w_bundle = w[ w_idx==WEIGHTS_out_idx[key] ].astype(np.float32)
                    if weights_in_ext=='.txt':
                        np.savetxt( os.path.join(output_folder,f'{pref_key}.txt'), w_bundle, fmt='%.5e' )
                    else:
                        np.save( os.path.join(output_folder,f'{pref_key}.npy'), w_bundle, allow_pickle=False )

        if len(regions)==0:
            if unassigned_count:
                logger.subinfo(f'Number of connecting: {n_written-TCK_outs_size["unassigned"]}', indent_char='*', indent_lvl=1)
                logger.subinfo(f'Number of non-connecting: {TCK_outs_size["unassigned"]}', indent_char='*', indent_lvl=1)
            else:
                logger.subinfo(f'Number of connecting: {n_written}', indent_char='*', indent_lvl=1)

    except Exception as e:
        if os.path.isdir(output_folder):
            for key in TCK_outs.keys():
                pref_key = f'{prefix}{key}'
                basename = os.path.join(output_folder,pref_key)
                if os.path.isfile(basename+'.tck'):
                    os.remove(basename+'.tck')
                if weights_in is not None and os.path.isfile(basename+weights_in_ext):
                    os.remove(basename+weights_in_ext)

        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        logger.subinfo('Closing files', indent_char='*', indent_lvl=1, with_progress=verbose>2)
        with ProgressBar(total=len(TCK_outs), disable=verbose < 3, hide_on_exit=True, subinfo=True) as pbar:
            if TCK_in is not None:
                TCK_in.close()
            for key in TCK_outs.keys():
                if key=='unassigned':
                    pref_key = 'unassigned'
                else:
                    pref_key = f'{prefix}{key}'
                f = os.path.join(output_folder,f'{pref_key}.tck')
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


def join( input_list: list[str], output_tractogram: str, weights_list: list[str]=[], weights_out: str=None, verbose: int=3, force: bool=False, log_list=None ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """
    set_verbose('tractogram', verbose)

    if len(input_list) < 2:
        logger.error(f'Input list must contain at least 2 files')
    files = [File(name=f'input_tractogram_{i}', type_='input', path=f) for i, f in enumerate(input_list)]
    files.append(File(name='output_tractogram', type_='output', path=output_tractogram, ext='.tck'))
    if weights_list:
        if len(input_list) != len(weights_list):
            logger.error(f'Number of weights files is different from number of input tractograms')
        for i, w in enumerate(weights_list):
            files.append(File(name=f'weights_in_{i}', type_='input', path=w, ext=['.txt', '.npy']))
    if weights_out is not None:
        files.append(File(name='weights_out', type_='output', path=weights_out, ext=['.txt', '.npy']))
    check_params(files=files, force=force)

    #----- iterate over input files -----
    logger.info('Joining tractograms')
    t0 = time()
    logger.subinfo(f'Output tractogram path: \'{output_tractogram}\'', indent_char='*', indent_lvl=1)
    TCK_in    = None
    TCK_out   = None
    n_written = 0
    weights_tot = np.array([], dtype=np.float32)
    try:
        # open the output file
        TCK_in = LazyTractogram( input_list[0], mode='r' )
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        TCK_in.close()

        with ProgressBar( total=len(input_list), disable=verbose < 3, hide_on_exit=True) as pbar:
            for i,input_tractogram in enumerate(input_list):

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

                if weights_list:
                    # load weights file
                    weights_in_ext = os.path.splitext(weights_list[i])[1]
                    if weights_in_ext == '.txt':
                        w = np.loadtxt(weights_list[i]).astype(np.float32)
                    elif weights_in_ext == '.npy':
                        w = np.load(weights_list[i], allow_pickle=False).astype(np.float64)
                    # check if #(weights)==n_streamlines
                    if n_streamlines!=w.size:
                        logger.error(f'# of weights {w.size} is different from # of streamlines ({n_streamlines}) in file {input_tractogram}')
                    # append weights
                    weights_tot = np.append(weights_tot, w)

                pbar.update()

            if weights_out is not None and weights_tot.size>0:
                logger.subinfo(f'Output weights path: \'{weights_out}\'', indent_char='*', indent_lvl=1)
                weights_out_ext = os.path.splitext(weights_out)[1]
                if weights_out_ext == '.txt':
                    np.savetxt(weights_out, weights_tot.astype(np.float32), fmt='%.5e')
                elif weights_out_ext == '.npy':
                    np.save(weights_out, weights_tot.astype(np.float32), allow_pickle=False)
                logger.subinfo(f'Total output weigths: {weights_tot.size}', indent_char='*', indent_lvl=1)

        logger.subinfo(f'Total output streamlines: {n_written}', indent_char='*', indent_lvl=1)
    except Exception as e:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( output_tractogram ):
            os.remove( output_tractogram )
        if weights_out is not None and os.path.isfile( weights_out ):
            os.remove( weights_out )
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )
    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( write_eof=True, count=n_written )
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


def sort(input_tractogram: str, input_atlas: str, output_tractogram: str=None, weights_in: str=None, weights_out: str=None, tmp_folder: str=None, keep_tmp_folder: bool=False, verbose: int=3, force: bool=False ):
    """Sort the streamlines in a tractogram bundle-by-bundle in lexigraphical order (i.e., bundle_1-1 --> bundle_1-2 --> ... --> bundle_2-2 --> ...).

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to sort.

    input_atlas : string
        Path to the file (.nii.gz) containing the gray matter parcellation.

    output_tractogram : string
        Path to the file where to store the sorted tractogram. If not specified (default),
        the new file will be created by appending '_sorted' to the input filename.

    weights_in : string
        Text file with the input streamline weights (one row/streamline).

    weights_out : str
        Scalar file (.txt or .npy) for the output streamline weights.

    tmp_folder : str
        Path to the temporary folder used to store the intermediate files.

    keep_tmp_folder : boolean
        Keep the temporary folder (default : False).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).

    """
    from dicelib.connectivity import assign #build_connectome

    set_verbose('tractogram', verbose)

    # check input files
    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='input_atlas', type_='input', path=input_atlas)
    ]
    if weights_in is not None:
        files.append(File(name='weights_in', type_='input', path=weights_in, ext=['.txt', '.npy']))
        weights_in_ext = os.path.splitext(weights_in)[1]
        if weights_out is not None:
            weights_out_ext = os.path.splitext(weights_out)[1]
            files.append(File(name='weights_out', type_='output', path=weights_out, ext=['.txt', '.npy']))
        else:
            weights_out = os.path.splitext(weights_in)[0]+f'_sorted.{weights_in_ext}'
            weights_out_ext = weights_in_ext
            files.append(File(name='weights_out', type_='output', path=weights_out, ext=['.txt', '.npy']))
    
    if output_tractogram is None:
        output_tractogram = os.path.splitext(input_tractogram)[0]+'_sorted.tck'
    files.append(File(name='output_tractogram', type_='output', path=output_tractogram, ext='.tck'))

    tmp_folder = tmp_folder if tmp_folder is not None else os.path.join(os.getcwd(), 'tmp_sort')
    dirs = [Dir(name='tmp_folder', path=tmp_folder)]
    check_params(files=files, dirs=dirs, force=force)

    tmp_dir_is_created = False
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        tmp_dir_is_created = True

    logger.info('Sorting tractogram')
    t0 = time()

    # compute assignments
    log_list_asgn = []
    ret_subinfo = logger.subinfo('Computing assignments', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo, log_list=log_list_asgn):
        assign(input_tractogram, input_atlas, assignments_out=f'{tmp_folder}/fibers_assignment.txt', verbose=1, force=force, log_list=log_list_asgn)

    # split the tractogram
    log_list_split = []
    ret_subinfo_split = logger.subinfo('Splitting tractogram', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_split, log_list=log_list_split):
        if weights_in is not None:
            split(input_tractogram, f'{tmp_folder}/fibers_assignment.txt', f'{tmp_folder}/bundles', weights_in=weights_in, verbose=1, force=force, log_list=log_list_split)
        else:
            split(input_tractogram, f'{tmp_folder}/fibers_assignment.txt', f'{tmp_folder}/bundles', verbose=1, force=force, log_list=log_list_split)
    set_verbose('tractogram', verbose)

    # join the tractograms
    asgn = np.loadtxt( f'{tmp_folder}/fibers_assignment.txt', dtype=np.int32 )
    max_rois = asgn.max()
    log_list_join = []
    ret_subinfo_join = logger.subinfo('Joining bundles in the specific order', indent_lvl=1, indent_char='*', with_progress=verbose>2)
    with ProgressBar(disable=verbose < 3, hide_on_exit=True, subinfo=ret_subinfo_join, log_list=log_list_join):
        list_all = []
        list_all_weights = []
        for i in range(max_rois):
            for j in range(i, max_rois):
                path_bundle = f'{tmp_folder}/bundles/bundle_{i+1}-{j+1}.tck'
                if os.path.isfile(path_bundle):
                    list_all.append(path_bundle)
                    if weights_in is not None:
                        path_weights = f'{tmp_folder}/bundles/bundle_{i+1}-{j+1}{weights_in_ext}'
                        list_all_weights.append(path_weights)
        if weights_in is not None:
            join(list_all, output_tractogram, weights_list=list_all_weights, weights_out=weights_out, verbose=1, log_list=log_list_join)
        else:
            join(list_all, output_tractogram, verbose=1, log_list=log_list_join)
    set_verbose('tractogram', verbose)
    if os.path.isfile(f'{tmp_folder}/bundles/unassigned.tck'):
        logger.warning('Some streamlines of the input tractogram are \'non-connecting\'')

    # remove temporary folder/files
    if not keep_tmp_folder:
        shutil.rmtree(f'{tmp_folder}/bundles')
        os.remove(f'{tmp_folder}/fibers_assignment.txt')
        # remove tmp_folder if different from current
        if tmp_dir_is_created:
            shutil.rmtree(tmp_folder)

    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


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


def sanitize(input_tractogram: str, gray_matter: str, white_matter: str, output_tractogram: str=None, step: float=0.2, max_dist: float=2, save_connecting_tck: bool=False, verbose: int=3, force: bool=False ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
     """

    set_verbose('tractogram', verbose)

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_sanitized'+extension
    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram, ext='.tck'),
        File(name='gray_matter', type_='input', path=gray_matter, ext=['.nii', '.nii.gz']),
        File(name='white_matter', type_='input', path=white_matter, ext=['.nii', '.nii.gz']),
        File(name='output_tractogram', type_='output', path=output_tractogram, ext='.tck')
    ]
    if save_connecting_tck == True :
        basename, extension = os.path.splitext(output_tractogram)
        conn_tractogram = basename+'_only_connecting'+extension
        files.append(File(name='conn_tractogram', type_='output', path=conn_tractogram, ext='.tck'))
    check_params(files=files, force=force)
    
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
        logger.error('Images have different shapes')

    if wm_header['pixdim'][1] != gm_header['pixdim'][1] or wm_header['pixdim'][2] != gm_header['pixdim'][2] or wm_header['pixdim'][3] != gm_header['pixdim'][3]:
        logger.error('Images have different pixel size')

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
    cdef float [:] moved_pt = np.zeros(3, dtype=np.float32)

    try:
        # open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        if n_streamlines == 0:
            logger.error('No streamlines found')

        # open the output file
        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )
        if save_connecting_tck==True:
            TCK_con = LazyTractogram( conn_tractogram, mode='w', header=TCK_in.header )

        logger.info('Tractogram sanitize')
        t0 = time()
        logger.subinfo(f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*', indent_lvl=1)

        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                n_pts_out = TCK_in.n_pts
                idx_last  = TCK_in.n_pts - 1

                fib = np.asarray(TCK_in.streamline)
                fib = fib[:TCK_in.n_pts, :]
                for n in xrange(3): # move first 3 point at each end
                    fib[n,:] = apply_affine_1pt(fib[n,:], M_inv, abc_inv, moved_pt)
                    fib[idx_last-n,:] = apply_affine_1pt( fib[idx_last-n,:], M_inv, abc_inv, moved_pt)
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
                    fib[n,:] = apply_affine_1pt( fib[n,:], M_dir, abc_dir, moved_pt)
                    fib[idx_last-n,:] = apply_affine_1pt( fib[idx_last-n,:], M_dir, abc_dir, moved_pt)
                if del_both[0] == False:    
                    fib[2,:] = apply_affine_1pt( fib[2,:], M_dir, abc_dir, moved_pt)
                if del_both[1] == False:
                    fib[idx_last-2,:] = apply_affine_1pt( fib[idx_last-2,:], M_dir, abc_dir, moved_pt) 

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
        t1 = time()
        

    logger.subinfo(f'Sanitized tractogram path: \'{output_tractogram}\'', indent_char='*', indent_lvl=1)
    if save_connecting_tck:
        logger.subinfo(f'Connecting streamlines path: \'{conn_tractogram}\'', indent_char='*', indent_lvl=1)
    logger.subinfo(f'Tot. streamlines: {n_tot}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'Connecting (both ends in GM): {n_in}', indent_lvl=2, indent_char='-')
    logger.subinfo(f'Half connecting (one ends in GM): {n_half}', indent_lvl=2, indent_char='-')
    logger.subinfo(f'Non-connecting (both ends outside GM): {n_out}', indent_lvl=2, indent_char='-')
    logger.info( f'[ {format_time(t1 - t0)} ]' )


def spline_smoothing_v2( input_tractogram, output_tractogram=None, spline_type='centripetal', epsilon=None, n_ctrl_pts=None, segment_len=None, streamline_pts=None, verbose=3, force=False ):
    """Smooth each streamline in the input tractogram using Catmull-Rom splines.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_smooth' to the input filename.

    spline_type : string
        Type of the Catmull-Rom spline: 'centripetal', 'uniform' or 'chordal' (default: 'centripetal').

    epsilon : float
        Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline (default: None).

    n_ctrl_pts : int
        Number of control points of the spline used by Ramer-Douglas-Peucker algorithm. NOTE: either 'epsilon' or 'n_ctrl_pts' must be set (default: None).

    segment_len : float
        Sampling resolution of the final streamline after interpolation. NOTE: either 'segment_len' or 'streamline_pts' must be set (default: None).

    streamline_pts : int
        Number of points in each of the final streamlines. NOTE: either 'streamline_pts' or 'segment_len' must be set (default: None).

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('tractogram', verbose)

    if segment_len is not None and streamline_pts is not None:
        logger.error('Either \'streamline_pts\' or \'segment_len\' must be set, not both.')
    if segment_len is None and streamline_pts is None:
        segment_len = 0.5
        streamline_pts = 0
    if segment_len is None:
        segment_len = 0
    if streamline_pts is None:
        streamline_pts = 0

    if epsilon is not None and n_ctrl_pts is not None:
        logger.error('Either \'epsilon\' or \'n_ctrl_pts\' must be set, not both.')
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
        logger.error(f'\'n_ctrl_pts\'must be an integer data type.')


    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_smooth'+extension

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='output_tractogram', type_='output', path=output_tractogram, ext=['.tck', '.trk'])
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

    try:
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

        logger.info('Smoothing tractogram')
        t0 = time()
        logger.subinfo(f'Input tractogram: {input_tractogram}', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=2, indent_char='-')

        mb = os.path.getsize( input_tractogram )/1.0E6
        if mb >= 1E3:
            logger.debug(f'Size: {mb/1.0E3:.2f} GB')
        else:
            logger.debug(f'Size: {mb:.2f} MB')

        if n_ctrl_pts != 0:
            logger.subinfo(f'Number of control points: {n_ctrl_pts}', indent_lvl=1, indent_char='*')
        if epsilon != 0:
            logger.subinfo(f'Epsilon for the control points reduction: {epsilon:.2f}', indent_lvl=1, indent_char='*')

        logger.subinfo(f'Output tractogram: {output_tractogram}', indent_char='*', indent_lvl=1)
        logger.subinfo(f'Spline type: {spline_type}', indent_lvl=2, indent_char='-')
        if segment_len != 0:
            logger.subinfo(f'Segment length: {segment_len:.2f}', indent_lvl=2, indent_char='-')
        if streamline_pts != 0:
            logger.subinfo(f'Number of final points: {streamline_pts}', indent_lvl=2, indent_char='-')

        # process each streamline
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                if segment_len != 0:
                    smoothed_streamline, n = apply_smoothing(TCK_in.streamline, TCK_in.n_pts, segment_len=segment_len, epsilon=epsilon, alpha=alpha, n_pts_red=n_ctrl_pts)
                if streamline_pts != 0:
                    smoothed_streamline, n = apply_smoothing(TCK_in.streamline, TCK_in.n_pts, n_pts_final=streamline_pts, epsilon=epsilon, alpha=alpha, n_pts_red=n_ctrl_pts)
                TCK_out.write_streamline( smoothed_streamline, n )
                pbar.update()

    except Exception as e:
        TCK_out.close()
        if os.path.exists( output_tractogram ):
            os.remove( output_tractogram )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        TCK_in.close()
        TCK_out.close()

    mb = os.path.getsize( output_tractogram )/1.0E6
    if mb >= 1E3:
        logger.debug(f'{mb/1.0E3:.2f} GB')
    else:
        logger.debug(f'{mb:.2f} MB')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef smooth_tractogram( input_tractogram, output_tractogram=None, mask=None, pts_cutoff=0.5, spline_type='centripetal', epsilon=0.3, segment_len=None, streamline_pts=None, verbose=3, force=False ):
    """Smooth each streamline in the input tractogram using Catmull-Rom splines.
    More info at http://algorithmist.net/docs/catmullrom.pdf.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_smooth' to the input filename.

    mask : string
        Path to the mask file (.nii) to constrain the smoothing to a specific region (default : None).

    pts_cutoff : float
        Percentage of points of the streamline that must be inside the mask to be considered (default : 0.5).

    spline_type : string
        Type of the Catmull-Rom spline: 'centripetal', 'uniform' or 'chordal' (default : 'centripetal').

    epsilon : float
        Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline (default : 0.3).

    segment_len : float
        Sampling resolution of the final streamline after interpolation. NOTE: either 'segment_len' or 'streamline_pts' must be set.

    streamline_pts : int
        Number of points in each of the final streamlines. NOTE: either 'streamline_pts' or 'segment_len' must be set.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('tractogram', verbose)

    if segment_len==None and streamline_pts==None:
        logger.error('Either \'streamline_pts\' or \'segment_len\' must be set.')
    if segment_len!=None and streamline_pts!=None:
        logger.error('Either \'streamline_pts\' or \'segment_len\' must be set, not both.')

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_smooth'+extension

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='output_tractogram', type_='output', path=output_tractogram, ext=['.tck', '.trk'])
    ]
    if mask is not None:
        files.append( {'type_': 'input', 'name': 'mask', 'path': mask} )
    nums = [
        Num(name='epsilon', value=epsilon, min_=0.0, max_=None,)
    ]
    check_params(files=files, nums=nums, force=force)

    if spline_type == 'centripetal':
        alpha = 0.5
    elif spline_type == 'chordal':
        alpha = 1.0
    elif spline_type == 'uniform':
        alpha = 0.0
    else:
        logger.error('\'spline_type\' parameter must be \'centripetal\', \'uniform\' or \'chordal\'')

    # if epsilon < 0 :
    #     raise ValueError( "'epsilon' parameter must be non-negative" )

    # cdef float [:,:] smoothed_fib = np.zeros((1000,3), dtype=np.float32)
    # cdef float [:,:] resampled_fib = np.zeros((1000,3), dtype=np.float32)
    cdef int n_pts_tot = 0
    cdef int n_pts_in = 0
    cdef int[:,:,:] mask_view
    cdef float[:] pt_aff = np.zeros(3, dtype=np.float32)
    cdef double [:,::1] mask_aff_inv 
    cdef double [::1,:] M_inv        
    cdef double [:] abc_inv
    cdef cbool in_mask
    cdef float fib_len = 0
    cdef int n_pts_out = 0
    cdef int in_mask_count
    cdef float epsilon_tmp
    cdef int n_pts_tmp = 0
    cdef size_t i, j = 0
    cdef int attempts = 0
    cdef float threshold = pts_cutoff

    if mask is not None:
        if not os.path.isfile(mask):
            logger.error(f'File \'mask\' not found')
        mask_nii = nib.load(mask)
        mask_view = np.ascontiguousarray(mask_nii.get_fdata(), dtype=np.int32)
        mask_aff_inv = np.linalg.inv(mask_nii.affine)
        M_inv = mask_aff_inv[:3, :3].T
        abc_inv = mask_aff_inv[:3, 3]

    try:
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

        logger.info('Smoothing tractogram')
        t0 = time()
        logger.subinfo('Input tractogram', indent_char='*', indent_lvl=1)
        logger.subinfo(f'{input_tractogram}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='-')

        mb = os.path.getsize( input_tractogram )/1.0E6
        if mb >= 1E3:
            logger.debug(f'{mb/1.0E3:.2f} GB', indent_lvl=1, indent_char='-')
        else:
            logger.debug(f'{mb:.2f} MB', indent_lvl=1, indent_char='-')

        logger.subinfo('Output tractogram', indent_char='*', indent_lvl=1)
        logger.subinfo(f'{output_tractogram}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Spline type: {spline_type}', indent_lvl=1, indent_char='-')
        if not segment_len==None:
            logger.subinfo(f'Segment length: {segment_len:.2f}', indent_lvl=1, indent_char='-')
        if not streamline_pts==None:
            logger.subinfo(f'Number of points: {streamline_pts}', indent_lvl=1, indent_char='-')
        if mask is not None:
            logger.subinfo(f'Mask: {mask}', indent_lvl=1, indent_char='-')

        if streamline_pts!=None:
            n_pts_out = streamline_pts
        else:
            n_pts_out = 50


        # process each streamline
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True ) as pbar:
            for i in range( n_streamlines ):
                epsilon_tmp = epsilon
                in_mask = False
                TCK_in.read_streamline()
                n_pts_in = TCK_in.n_pts
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                while in_mask==False:
                    # smoothed_streamline, n = apply_smoothing(TCK_in.streamline, TCK_in.n_pts, segment_len=segment_len, epsilon=epsilon, alpha=alpha)
                    fib_red_ptr, n_red = rdp_reduction(TCK_in.streamline, n_pts_in, epsilon_tmp)

                    # check number of points 
                    if n_red==2: # no need to smooth
                        smoothed_fib = fib_red_ptr
                        n_pts_tot = n_red
                        in_mask = True
                    else:
                        smoothed_fib =  apply_smoothing(fib_red_ptr, alpha, n_pts_out)
                        in_mask_count = 0 
                        for j in range(n_pts_out):
                            pt_aff = apply_affine_1pt(smoothed_fib[j,:], M_inv, abc_inv, pt_aff)
                            if mask_view[<int>(pt_aff[0]+0.5), <int>(pt_aff[1]+0.5), <int>(pt_aff[2]+0.5)] > 0:
                                in_mask_count += 1
                        if in_mask_count > threshold*n_pts_out:
                            in_mask = True
                        else:
                            # reduce epsilon and try again
                            attempts += 1
                            epsilon_tmp = epsilon_tmp-0.1
                            if epsilon_tmp < 0.1:
                                smoothed_fib = fib_red_ptr
                                n_pts_tot = n_pts_in
                                in_mask = True

                    # compute streamline length
                    fib_len = streamline_length( smoothed_fib, n_pts_out )

                    if segment_len!=None:
                        n_pts_out = int(fib_len / segment_len)


                    # resample smoothed streamline
                    resampled_fib = s_resample(smoothed_fib, n_pts_out)

                TCK_out.write_streamline( resampled_fib, n_pts_out )
                pbar.update()

    except Exception as e:
        TCK_out.close()
        if os.path.exists( output_tractogram ):
            os.remove( output_tractogram )
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        TCK_in.close()
        TCK_out.close()

    mb = os.path.getsize( output_tractogram )/1.0E6
    logger.debug(f'{mb:.2f} MB', indent_lvl=1, indent_char='-')
    if mb >= 1E3:
        logger.debug(f'{mb/1.0E3:.2f} GB', indent_lvl=1, indent_char='-')
    else:
        logger.debug(f'{mb:.2f} MB', indent_lvl=1, indent_char='-')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )



cpdef spline_smoothing( input_tractogram, output_tractogram=None, control_point_ratio=0.25, segment_len=1.0, verbose=3, force=False ):
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
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """

    set_verbose('tractogram', verbose)

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_smooth'+extension
    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='output_tractogram', type_='output', path=output_tractogram, ext=['.tck', '.trk'])
    ]
    nums = [
        Num(name='control_point_ratio', value=control_point_ratio, min_=0.0, max_=1.0, include_min=False),
        Num(name='segment_len', value=segment_len, min_=0.0, max_=None, include_min=False)
    ]
    check_params(files=files, nums=nums, force=force)

    try:
        TCK_in = LazyTractogram( input_tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )

        TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

        logger.info('Smoothing tractogram')
        t0 = time()
        logger.subinfo('Input tractogram', indent_char='*', indent_lvl=1)
        logger.subinfo(f'{input_tractogram}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='-')

        mb = os.path.getsize( input_tractogram )/1.0E6
        if mb >= 1E3:
            logger.debug(f'{mb/1.0E3:.2f} GB', indent_lvl=1, indent_char='-')
        else:
            logger.debug(f'{mb:.2f} MB', indent_lvl=1, indent_char='-')

        logger.subinfo('Output tractogram', indent_char='*', indent_lvl=1)
        logger.subinfo(f'{output_tractogram}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Control points: {control_point_ratio*100.0:.1f}%', indent_lvl=1, indent_char='-')
        logger.subinfo(f'Segment length: {segment_len:.2f}', indent_lvl=1, indent_char='-')

        # process each streamline
        with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
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
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')

    finally:
        TCK_in.close()
        TCK_out.close()

    mb = os.path.getsize( output_tractogram )/1.0E6
    if mb >= 1E3:
        logger.debug(f'{mb/1.0E3:.2f} GB', indent_lvl=1, indent_char='-')
    else:
        logger.debug(f'{mb:.2f} MB', indent_lvl=1, indent_char='-')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )


def recompute_indices(input_indices, dictionary_kept, output_indices=None, verbose=3, force=False):
    """Recompute the indices of the streamlines in a tractogram after filtering.

    Parameters
    ----------
    input_indices : array of integers
        Indices of the streamlines in the original tractogram.

    dictionary_kept : dictionary
        Dictionary of the streamlines kept after filtering.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    Returns
    -------
    indices_recomputed : array of integers
        Recomputed indices of the streamlines.
    """
    set_verbose('tractogram', verbose)

    files = [
        File(name='input_indices', type_='input', path=input_indices),
        File(name='dictionary_kept', type_='input', path=dictionary_kept)
    ]
    if output_indices is not None:
        files.append( File(name='output_indices', type_='output', path=output_indices) )
    check_params(files=files, force=force)

    # open indices file and dictionary
    d = np.fromfile(dictionary_kept, dtype=np.uint8)

    idx = np.loadtxt(input_indices).astype(np.uint32)
    indices_recomputed = []

    # recompute indices
    logger.info('Recomputing indices')
    t0 = time()
    with ProgressBar( total=idx.size, disable=verbose < 3, hide_on_exit=True) as pbar:
        for i in range( idx.size ):
            #count the number of streamlines before the current one
            n = np.count_nonzero( d[:idx[i]] )

            # check if the current streamline is kept
            if d[idx[i]]==1:
                indices_recomputed.append( n )
            pbar.update()
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
    return indices_recomputed if output_indices is None else np.savetxt(output_indices, indices_recomputed, fmt='%d')


cpdef sample(input_tractogram, input_image, output_file, mask_file=None, option="No_opt", collapse=False, force=False, verbose=3):
    """Sample underlying values of a tractogram along its points from the corresponding image (ATTENTION: this method does not use interpolation during sampling)

    Parameters
    ----------
    input_tractogram : string 
        Path to the file (.tck) containing the streamlines to process.
    input_image : string 
        Path to the image where the method has to sample values.
    output_file : string 
        Path to the file (.txt is needed) where the method saves values
    mask_file : string (default None)
        Path to the mask file (.nii) to constrain the sampling to a specific region.
    option : string (default None)
        apply some operation on values
    collpase : boolean
        If True, the method will collapse the values of points falling in the same voxel (default : False).
    force : boolean
        Force overwriting of the output (default : False).
    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    Returns
    -------
    Text file with values of input tractogram in the referred input image. 


    """
    set_verbose('tractogram', verbose)

    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='input_image', type_='input', path=input_image),
        File(name='output_file', type_='output', path=output_file)
    ]
    if mask_file is not None:
        files.append(File(name='mask_file', type_='input', path=mask_file))
    if option not in ['min', 'max', 'median', 'No_opt', 'mean']:
        logger.error(f'Option {option} not valid, please choose between min, max, median, mean or No_opt')
    check_params(files=files, force=force)

    TCK_in  = None

    #open the image
    Img = nib.load(input_image)
    img_data = Img.get_fdata()
    img_data = np.array(img_data, dtype=np.float32)
    

    if mask_file != None:
        #open the mask
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata()
    else:
        mask_data = np.ones(img_data.shape, dtype=np.float32)

    cdef float [:,:,::1] img_view = np.ascontiguousarray(img_data).astype(np.float32)
    cdef float [:,:,::1] mask_view = np.ascontiguousarray(mask_data).astype(np.float32)
    cdef double [:,::1] wm_aff_inv  = np.linalg.inv(Img.affine) #inverse of affine
    cdef double [::1,:] M_inv       = wm_aff_inv[:3, :3].T 
    cdef double [:] abc_inv         = wm_aff_inv[:3, 3]
    cdef float [:] moved_pt         = np.zeros(3, dtype=np.float32)
    cdef size_t ii                  = 0
    cdef size_t jj                  = 0
    cdef int [::1] vox_coords       = np.zeros(3, dtype=np.int32)
    cdef float [::1] value          = np.zeros(2000, dtype=np.float32)
    cdef int [:,::1] voxel_checked
    cdef int tot_vox                = 0
    
    try:
        #open the input file
        TCK_in = LazyTractogram( input_tractogram, mode='r' )

        n_streamlines = int( TCK_in.header['count'] )
        logger.info(f'Tractogram sampling')
        t0 = time()
        logger.subinfo(f'Number of streamlines in input tractogram: {n_streamlines}', indent_char='*', indent_lvl=1)

        pixdim = Img.header['pixdim'] [1:4] 
        logger.subinfo('Image resolution: {}'.format(pixdim), indent_char='*', indent_lvl=1)
        logger.subinfo('Applying vox transformation and sampling values', indent_char='*', indent_lvl=1)

        with open(output_file,'w') as file:
            file.write("# dicelib.tractogram.sample option={} {} {} {}**\n".format(option,input_tractogram,input_image,output_file))
            with ProgressBar( total=n_streamlines, disable=verbose<3, hide_on_exit=True) as pbar:
                for i in range(n_streamlines):
                    tot_vox = 0
                    TCK_in.read_streamline()
                    npoints = TCK_in.n_pts
                    voxel_checked = np.zeros((npoints,3), dtype=np.int32)
                    value = np.zeros(2000, dtype=np.float32)
                    for ii in range(npoints):
                        moved_pt = apply_affine_1pt(TCK_in.streamline[ii], M_inv, abc_inv, moved_pt)
                        vox_coords[0] = int(moved_pt[0])
                        vox_coords[1] = int(moved_pt[1])
                        vox_coords[2] = int(moved_pt[2])
                        if collapse:
                            # check if the voxel has already been visited
                            for jj in range(ii):
                                if voxel_checked[jj,0] == vox_coords[0] and voxel_checked[jj,1] == vox_coords[1] and voxel_checked[jj,2] == vox_coords[2]:
                                    break
                            if jj < ii-1:
                                continue
                            else:
                                tot_vox += 1
                                voxel_checked[tot_vox] = vox_coords
                            npoints = tot_vox

                        if mask_view[vox_coords[0], vox_coords[1], vox_coords[2]] == 0:
                            value[ii] = np.nan
                        else: 
                            value[ii] = img_view[vox_coords[0], vox_coords[1], vox_coords[2]]

                    if option == 'No_opt':
                        np.savetxt(file, value[:npoints], fmt='%.3f', newline=' ')
                        file.write("\n")
                    elif option == 'mean':
                        value[ii+2] = np.nanmean(value[:ii+1])
                        file.write(f'{value[ii+2]:.3f}')
                        file.write("\n")
                    elif option == 'median':
                        value[ii+3] = np.nanmedian(value[:ii+1])
                        file.write(f'{value[ii+3]:.3f}')
                        file.write("\n")
                    elif option == 'min':
                        value[ii+4] = np.min(value[:ii+1])
                        file.write(f'{value[ii+4]:.3f}')
                        file.write("\n")
                    elif option == 'max':
                        value[ii+5] = np.max(value[:ii+1])
                        file.write(f'{value[ii+5]:.3f}')
                        file.write("\n")

                    pbar.update()


    except Exception as e:
        if TCK_in is not None:
            TCK_in.close()
            logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')
    finally:
        if TCK_in is not None:
            TCK_in.close()
        file.close()
        t1 = time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )


cpdef resample(input_tractogram, output_tractogram, nb_pts, verbose=3, force=False):
    """Set the number of points of each streamline in the input tractogram.

    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.

    output_tractogram : string
        Path to the file where to store the filtered tractogram. If not specified (default),
        the new file will be created by appending '_nbpts' to the input filename.

    nb_pts : int
        Number of points to set for each streamline.

    verbose : int
        What information to print, must be in [0...4] as defined in ui.set_verbose() (default : 3).

    force : boolean
        Force overwriting of the output (default : False).
    """
    set_verbose('tractogram', verbose)

    if output_tractogram is None :
        basename, extension = os.path.splitext(input_tractogram)
        output_tractogram = basename+'_nbpts'+extension
    files = [
        File(name='input_tractogram', type_='input', path=input_tractogram),
        File(name='output_tractogram', type_='output', path=output_tractogram)
    ]
    nums = [Num(name='nb_pts', value=nb_pts, min_=2)]
    check_params(files=files, nums=nums, force=force)

    cdef float [::1] lengths = np.empty( 1000, dtype=np.float32 )
    cdef float [:,::1] s0 = np.empty( (nb_pts, 3), dtype=np.float32 )
    cdef float [::1] vers = np.empty( 3, dtype=np.float32 )

    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    n_streamlines = int( TCK_in.header['count'] )

    TCK_out = LazyTractogram( output_tractogram, mode='w', header=TCK_in.header )

    logger.info('Resampling')
    t0 = time()
    logger.subinfo(f'Input tractogram: {input_tractogram}', indent_char='*', indent_lvl=1)
    logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=1, indent_char='*')

    mb = os.path.getsize( input_tractogram )/1.0E6
    if mb >= 1E3:
        logger.debug(f'{mb/1.0E3:.2f} GB')
    else:
        logger.debug(f'{mb:.2f} MB')

    logger.subinfo(f'Number of points: {nb_pts}', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Output tractogram: {output_tractogram}', indent_char='*', indent_lvl=1)

    # process each streamline
    with ProgressBar( total=n_streamlines, disable=verbose < 3, hide_on_exit=True) as pbar:
        for i in range( n_streamlines ):
            TCK_in.read_streamline() 
            set_number_of_points(TCK_in.streamline[:TCK_in.n_pts], nb_pts, s0, vers, lengths)
            TCK_out.write_streamline( s0, nb_pts )
            pbar.update()

    TCK_in.close()
    TCK_out.close()

    mb = os.path.getsize( output_tractogram )/1.0E6
    if mb >= 1E3:
        logger.debug( f'{mb/1.0E3:.2f} GB')
    else:
        logger.debug( f'{mb:.2f} MB')
    t1 = time()
    logger.info( f'[ {format_time(t1 - t0)} ]' )
