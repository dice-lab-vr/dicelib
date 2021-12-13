#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False
cimport cython
import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport fopen, fclose, FILE, fseek, SEEK_END, SEEK_SET, SEEK_CUR
from libc.stdio cimport fgets, fread, fwrite, perror
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp, strncmp, strchr, strlen
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libc.math cimport isnan, isinf, NAN

cdef float[3] NAN3 = {NAN, NAN, NAN}


cdef class LazyTCK:
    """Class to read/write streamlines from a .TCK file one by one.

    A tractogram can e opened in three different modalities:
    - 'r': reading
    - 'w': writing
    - 'a': appending
    TODO: complete this description
    """
    cdef public     str                             filename
    cdef public     unordered_map[string, string]   header
    cdef            bint                            is_open
    cdef            str                             mode
    cdef            int                             max_points
    cdef public     float[:,:]                      streamline
    cdef public     unsigned int                    n_pts
    cdef            FILE*                           fp
    cdef            float*                          buffer
    cdef            float*                          buffer_ptr
    cdef            float*                          buffer_end


    def __init__( self, char *filename, char* mode, header=None, unsigned int max_points=3000 ):
        """Initialize the class.

        Parameters
        ----------
        filename : string
            File name of the tractogram to open.
        mode : string
            Opens the tractogram for reading ('r'), writing ('w') or appending ('a') streamlines.
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header; this parameter is only required
            when writing streamlines to disk (default : None).
        max_points : unsigned int
            The maximum number of points/coordinates allowed for a streamline (default : 3000).
        """
        self.filename = filename
        self.is_open = False

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

        # print( f'-----> Open {self.filename} as "{self.mode}"' )
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


    cpdef read_streamline( self ):
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
        if streamline.ndim!=2 or streamline.shape[1]!=3:
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
                self.header[b'count'] = '%0*d' % (len(self.header[b'count']), count) # NB: use same number of characters
                self._write_header( self.header )

        self.is_open = False
        fclose( self.fp )
        self.fp = NULL


    cpdef _read_header( self ):
        """Read the header from file.
        After the reading, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to read streamlines.
        """
        cdef char[1000] line
        cdef char*      ptr
        cdef int        nLines = 0

        # print( f'[_read_header()] Reading header from "{self.filename}"' )

        if self.header.size() > 0:
            raise RuntimeError( 'Header already read' )

        fseek( self.fp, 0, SEEK_SET )

        # check if it's a valid TCK file
        if fgets( line, sizeof(line), self.fp )==NULL:
            print( self.mode )
            raise IOError( 'Problems reading header from file FIRST LINE' )
        line[strlen(line)-1] = 0
        if strncmp( line,'mrtrix tracks', 13)!=0:
            raise IOError( f'"{self.filename}" is not a valid TCK file' )

        # parse one line at a time
        while True:
            if nLines>=1000:
                raise RuntimeError( 'Problem parsing the header; too many header lines' )
            if fgets( line, sizeof(line), self.fp )==NULL:
                raise IOError( 'Problems reading header from file' )
            line[strlen(line)-1] = 0
            if strcmp(line,'END')==0:
                break
            ptr = strchr(line, ord(':'))
            if ptr==NULL:
                raise RuntimeError( 'Problem parsing the header; format not valid' )
            pos = ptr-line
            self.header[ line[:pos] ] = ptr+2
            nLines += 1

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if self.header.count( b'count' )==0:
            raise RuntimeError( 'Problem parsing the header; field "count" not found' )

        # check if datatype is 'Float32LE'
        if self.header.count( b'datatype' )==0:
            raise RuntimeError( 'Problem parsing the header; field "datatype" not found' )
        if self.header[b'datatype']!=b'Float32LE':
            raise RuntimeError( 'Unable to process file, as datatype "Float32LE" is not yet handled' )

        # move file pointer to beginning of binary data
        if self.header.count( b'file' )==0:
            raise RuntimeError( 'Problem parsing the header; field "file" not found' )
        fseek(self.fp, int( self.header[b'file'].substr(2) ), SEEK_SET)


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

        fseek( self.fp, 0, SEEK_SET )
        line = b'mrtrix tracks\n'
        fwrite( line.c_str(), 1, line.size(), self.fp )

        for key in header:
            if key=='file':
                continue
            if key=='count':
                header['count'] = header['count'].zfill(10) # ensure 10 digits are written
            line = f'{key}: {header[key]}\n'
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
        if self.mode=='r':
            free( self.buffer )
        fclose( self.fp )