# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False


import os, time
import numpy as np

from dicelib.streamline import rdp_reduction, CatmullRom_smooth, length, resample

from libc.stdio cimport fclose, fgets, FILE, fopen, fread, fseek, fwrite, SEEK_END, SEEK_SET
from libc.stdlib cimport malloc
from libc.string cimport strchr, strlen, strncmp
from libcpp.string cimport string
from dicelib.tractogram import LazyTractogram
from dicelib.blur import Blur
from dicelib.header import Header

cdef extern from "float16_float32_encode_decode.hpp":
    float float16_to_float32(const unsigned short value)

cdef extern from "float16_float32_encode_decode.hpp":
    unsigned short float32_to_float16(const float value)

cdef float[:, :] matrix = np.array([
    [2, -2, 1, 1],
    [-3, 3, -2, -1],
    [0, 0, 1, 0],
    [1, 0, 0, 0]
]).astype(np.float32)

cdef class Tcz:
    """Class to read/write tcz files for visualization.

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
    cdef readonly   float[:,::1]                    streamline
    cdef readonly   unsigned int                    max_points
    cdef            unsigned int                    n_pts
    cdef            FILE *                          fp
    cdef            unsigned short int *            buffer
    cdef            unsigned short int *            buffer_ptr
    cdef            unsigned short int *            buffer_end

    def __init__(self, char *filename, char *mode, header=None, unsigned int max_points=3000):
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
        _, self.suffix = os.path.splitext(filename)
        if self.suffix not in ['.tcz']:
            raise ValueError('Only ".tcz" files are supported for now.')

        if mode not in ['r', 'w', 'a']:
            raise ValueError('"mode" must be either "r", "w" or "a"')

        self.mode = mode
        self.streamline = None
        self.n_pts = 0
        self.buffer = NULL
        self.buffer_ptr = NULL
        self.buffer_end = NULL

        # open the file
        self.fp = fopen(self.filename, ('r+' if self.mode == 'a' else self.mode) + 'b')
        if self.fp == NULL:
            raise FileNotFoundError(f'Unable to open file: "{self.filename}"')

        self.header = {}
        if self.mode == 'r':

            self.buffer = <unsigned short int *> malloc(3 * 1000000 * sizeof(unsigned short int))

            if max_points <= 0:
                raise ValueError('"max_points" should be positive')
            self.max_points = max_points
            self.streamline = np.empty((self.max_points, 3), dtype=np.float32)

            # file is open for reading => need to read the header from disk
            self.header.clear()
            self._read_header()

        elif self.mode == 'w':
            # file is open for writing => need to write a header to disk
            self._write_header(header)

        else:
            # file is open for appending => move pointer to end
            fseek(self.fp, 0, SEEK_END)

        self.is_open = True

    cpdef _read_header(self):
        """
        Read the header from file.
        After the reading, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file.
        """
        cdef char[5000000] line  # a field can be max 5MB long
        cdef char *         ptr
        cdef int           nLines = 0

        if len(self.header) > 0:
            raise RuntimeError('Header already read')

        fseek(self.fp, 0, SEEK_SET)

        # check if it's a valid tcz file
        if fgets(line, sizeof(line), self.fp) == NULL:
            raise IOError('Problems reading header from file FIRST LINE')

        if strncmp(line, 'mrtrix tracks', 13) != 0:
            raise IOError(f'"{self.filename}" is not a valid tcz file')

        # parse one line at a time
        while True:
            if nLines >= 1000:
                raise RuntimeError('Problem parsing the header; too many header lines')
            if fgets(line, sizeof(line), self.fp) == NULL:
                raise IOError('Problems reading header from file')
            line[strlen(line) - 1] = 0
            if strncmp(line, 'END', 3) == 0:
                break
            ptr = strchr(line, ord(':'))
            if ptr == NULL:
                raise RuntimeError('Problem parsing the header; format not valid')
            key = str(line[:(ptr - line)])
            val = ptr + 2
            if key not in self.header:
                self.header[key] = val
            else:
                if type(self.header[key]) != list:
                    self.header[key] = [self.header[key]]
                self.header[key].append(val)
            nLines += 1

        blur = Blur.from_header(self.header)
        self.header['blur_core_extent'] = blur.core_extent
        self.header['blur_gauss_extent'] = blur.gauss_extent
        self.header['blur_spacing'] = blur.spacing
        self.header['blur_gauss_min'] = blur.gauss_min

        self.header['datatype'] = 'Float16'

        if 'epsilon' not in self.header:
            self.header['epsilon'] = 0.3
        else:
            self.header['epsilon'] = float(self.header['epsilon'])

        if 'segment_len' not in self.header:
            self.header['segment_len'] = 0.5
        else:
            self.header['segment_len'] = float(self.header['segment_len'])

        if 'representation' not in self.header:
            self.header['representation'] = 'polyline'

        if self.header['representation'] not in ['polyline', 'spline', 'rdp']:
            raise RuntimeError('Problem parsing the header; field "representation" is not a valid value')

        if 'count' not in self.header:
            raise RuntimeError('Problem parsing the header; field "count" not found')

        if type(self.header['count']) is list:
            raise RuntimeError('Problem parsing the header; field "count" has multiple values')

        if 'file' not in self.header:
            raise RuntimeError('Problem parsing the header; field "file" not found')

        if type(self.header['file']) is list:
            raise RuntimeError('Problem parsing the header; field "file" has multiple values')

        fseek(self.fp, int(self.header['file'][2:]), SEEK_SET)

    cpdef _write_header(self, header):
        """Write the header to file.
        After writing the header, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to write scalars.

        Parameters
        ----------
        header : dictionary
            A dictionary of 'key: value' pairs that define the items in the header.
        """
        cdef string line
        cdef int offset = 18  # accounts for 'mrtrix tracks\n' and 'END\n'

        if header is None or type(header) != dict:
            raise RuntimeError('Provided header is empty or invalid')

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if 'count' not in header:
            raise RuntimeError('Problem parsing the header; field "count" not found')
        if type(header['count']) == list:
            raise RuntimeError('Problem parsing the header; field "count" has multiple values')

        fseek(self.fp, 0, SEEK_SET)
        line = b'mrtrix tracks\n'
        fwrite(line.c_str(), 1, line.size(), self.fp)

        for key, val in header.items():
            if key == 'file':
                continue
            if key == 'count':
                val = header['count'] = header['count'].zfill(10)  # ensure 10 digits are written

            if type(val) == str:
                val = [val]
            for v in val:
                line = f'{key}: {v}\n'
                fwrite(line.c_str(), 1, line.size(), self.fp)
                offset += line.size()

        if "timestamp" not in header:
            line = f'timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
            fwrite(line.c_str(), 1, line.size(), self.fp)
            offset += line.size()

        line = f'{offset + 9:.0f}'
        line = f'file: . {offset + 9 + line.size():.0f}\n'
        fwrite(line.c_str(), 1, line.size(), self.fp)
        offset += line.size()

        line = b'END\n'
        fwrite(line.c_str(), 1, line.size(), self.fp)

        self.header = header.copy()

        # move file pointer to beginning of binary data
        fseek(self.fp, offset, SEEK_SET)

    cpdef write_streamline( self, float[:,:] streamline, unsigned short int n=65000 ):
        """Write a streamline at the current position in the file.

        Parameters
        ----------
        streamline : Nx3 numpy array
            The streamline data
        n : int
            Writes first n points of the streamline. If n == 65000 (default), writes all points.
            NB: be careful because, for efficiency, a streamline is represented as a fixed-size array
        """

        if  streamline.ndim != 2 or streamline.shape[1] != 3:
            raise RuntimeError( '"streamline" must be a Nx3 array' )
        if n == 65000:
            n = streamline.shape[0]
        if n == 0:
            return

        if not self.is_open:
            raise RuntimeError( 'File is not open' )
        if self.mode == 'r':
            raise RuntimeError( 'File is not open for writing/appending' )

        if self.header['representation'] in ['spline', 'rdp']:
            streamline, n = rdp_reduction(streamline, n, float(self.header['epsilon']))

        fwrite( <void *> &n, sizeof(unsigned short int), 1, self.fp)

        compressed_streamline = self.compress_streamline(streamline, n)
        if fwrite( &compressed_streamline[0,0], sizeof(unsigned short int), 3 * n, self.fp ) != 3 * n:
            raise IOError( 'Problems writing streamline data to file' )


    cpdef read_streamline(self):
        """
        Read next streamline from the current position in the file.

        For efficiency reasons, multiple streamlines are simultaneously loaded from disk using a buffer.
        The current streamline is stored in the fixed-size numpy array 'self.streamline' and its actual
        length, i.e., number of points/coordinates, is stored in 'self.n_pts'.

        Returns
        -------
        output : int
            Number of points/coordinates read from disk.
        """
        cdef float fib_len
        cdef float * ptr = &self.streamline[0, 0]
        cdef int    n_read

        if not self.is_open:
            raise RuntimeError('File is not open')
        if self.mode != 'r':
            raise RuntimeError('File is not open for reading')

        fread( <void*> &self.n_pts, sizeof(unsigned short int), 1, self.fp)

        for i in range(self.n_pts):
           if self.n_pts > self.max_points:
               raise RuntimeError(f'Problem reading data, streamline seems too long (>{self.max_points} points)')

           if self.buffer_ptr == self.buffer_end:  # reached end of buffer, need to reload
               n_read = fread(self.buffer, sizeof(unsigned short int), 3 * self.n_pts, self.fp)
               self.buffer_ptr = self.buffer
               self.buffer_end = self.buffer_ptr + n_read

           # copy coordinate from 'buffer' to 'streamline'
           ptr[0] = float16_to_float32(self.buffer_ptr[0])
           ptr[1] = float16_to_float32(self.buffer_ptr[1])
           ptr[2] = float16_to_float32(self.buffer_ptr[2])

           self.buffer_ptr += 3

           ptr += 3

        if self.header['representation'] == 'spline':
            if self.n_pts > 2:
                smoothed_streamline = np.asarray(CatmullRom_smooth(self.streamline[:self.n_pts,:], matrix, 0.5, 50))
                fib_len = length(smoothed_streamline, self.n_pts)

                if float(self.header['segment_len']) != 0:
                    self.n_pts = int(fib_len / float(self.header['segment_len']))

                self.streamline = resample(smoothed_streamline, self.n_pts)

            else: # no need to smooth with two points only, as we have only one line with two points
                self.streamline = self.streamline[:self.n_pts, :]

        else:
            self.streamline = self.streamline[:self.n_pts,:]

        return self.n_pts, self.streamline

    cpdef unsigned short int[:,:] compress_streamline(self, float[:,:] streamline, unsigned short int n):
        """
        It casts a streamline from float32 to float16

        Parameters
        ----------
        streamline : Nx3 numpy array
            The streamline data
            
        n: unsigned short int
            The number of points of the streamline
        """

        cdef unsigned short int[:,:] compressed_streamline = np.empty((n, 3), dtype=np.uint16)
        for i in range(n):
            for j in range(3):
                compressed_streamline[i][j] = float32_to_float16(streamline[i][j])

        return compressed_streamline

    cpdef close(self, int count=-1):
        """Close the file associated with the tcz.

        Parameters
        ----------
        count : int
            Update the 'count' field in the header with this value (default : -1, i.e. do not update)
        """

        if not self.is_open:
            return

        if self.mode != 'r':
            # update 'count' in header
            if count >= 0:
                if self.mode == 'a':
                    # in append mode the header is not read by default
                    self.header.clear()
                    self._read_header()
                self.header['count'] = '%0*d' % (len(self.header['count']), count)  # NB: use same number of characters
                self._write_header(self.header)

        self.is_open = False
        fclose(self.fp)
        self.fp = NULL

    def __dealloc__(self):
        if self.is_open:
            fclose(self.fp)

cdef class FileConverter:
    cpdef from_tck_to_tcz(self, char * filename_in, char * filename_out, header=None):
        _, suffix = os.path.splitext(filename_in)

        if suffix not in ['.tck']:
            raise ValueError('input file is not a valid ".tck"')
        _, suffix = os.path.splitext(filename_out)

        if suffix not in ['.tcz']:
            raise ValueError('output file is not a valid ".tcz"')

        tck_in = LazyTractogram(filename_in, 'r')

        tcz_header = header
        tcz_header['datatype'] = 'Float16'
        tcz_header['count'] = tck_in.header['count']
        number_of_streamlines = int(tcz_header['count'])

        tcz_out = Tcz(filename_out, 'w', tcz_header)

        for i in range(number_of_streamlines):
            n_points = tck_in.read_streamline()
            tcz_out.write_streamline(tck_in.streamline, n_points)

        tck_in.close()
        tcz_out.close()


    cpdef from_tcz_to_tck(self, char * filename_in, char * filename_out, header=None):
        _, suffix = os.path.splitext(filename_in)

        if suffix not in ['.tcz']:
            raise ValueError('output file is not a valid ".tcz"')
        _, suffix = os.path.splitext(filename_out)

        if suffix not in ['.tck']:
            raise ValueError('input file is not a valid ".tck"')

        tcz_in = Tcz(filename_in, 'r')

        tck_header = header
        tck_header['datatype'] = 'Float16'
        tck_header['count'] = tcz_in.header['count']
        number_of_streamlines = int(tck_header['count'])

        if 'representation' in tck_header: del tck_header['representation']
        if 'segment_len' in tck_header: del tck_header['segment_len']
        if 'epsilon' in tck_header: del tck_header['epsilon']

        tck_out = LazyTractogram(filename_out, 'w',)

        for i in range(number_of_streamlines):
            n_points, streamline = tcz_in.read_streamline()
            tck_out.write_streamline(streamline, n_points)

        tcz_in.close()
        tck_out.close()
