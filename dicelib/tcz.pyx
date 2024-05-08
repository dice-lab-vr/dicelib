# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False


import os, time

import numpy as np
from libc.stdio cimport fclose, fgets, FILE, fopen, fread, fseek, fwrite, SEEK_END, SEEK_SET
from libc.stdlib cimport malloc
from libc.string cimport strchr, strlen, strncmp
from libcpp.string cimport string

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
    cdef readonly                                   streamline
    cdef readonly   unsigned int                    n_pts
    cdef            FILE *                           fp
    cdef            float *                          buffer
    cdef            float *                          buffer_ptr
    cdef            float *                          buffer_end

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

        if self.mode == 'r':
            self.buffer = <float *> malloc(3 * 1000000 * sizeof(float))
            pass
        else:
            self.buffer = NULL

        self.n_pts = 0
        self.buffer_ptr = NULL
        self.buffer_end = NULL

        # open the file
        self.fp = fopen(self.filename, ('r+' if self.mode == 'a' else self.mode) + 'b')
        if self.fp == NULL:
            raise FileNotFoundError(f'Unable to open file: "{self.filename}"')

        self.header = {}
        if self.mode == 'r':
            # file is open for reading => need to read the header from disk
            self.header.clear()
            self._read_header()

            if self.header['streamline_representation'] == 'polyline':
                self.streamline = np.empty((max_points, 3), dtype=np.float16)
            else:
                self.streamline = np.empty((max_points, 3), dtype=np.float32)

        elif self.mode == 'w':
            # file is open for writing => need to write a header to disk
            self._write_header(header)
        else:
            # file is open for appending => move pointer to end
            fseek(self.fp, 0, SEEK_END)

        self.is_open = True

    cpdef _read_header(self):
        """Read the header from file.
        After the reading, the file pointer is located at the end of it, i.e., beginning of
        the binary data part of the file, ready to read scalars.
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

        # blur params
        if 'blur_core_extent' not in self.header:
            raise RuntimeError('Problem parsing the header; field "blur_core_extent" not found')
        if 'blur_gauss_extent' not in self.header:
            raise RuntimeError('Problem parsing the header; field "blur_gauss_extent" not found')
        if 'blur_spacing' not in self.header:
            raise RuntimeError('Problem parsing the header; field "blur_spacing" not found')
        if 'blur_gauss_min' not in self.header:
            raise RuntimeError('Problem parsing the header; field "blur_gauss_min" not found')

        if float(self.header['blur_core_extent']) < 0:
            raise RuntimeError('"blur_core_extent" must be >= 0')
        if float(self.header['blur_gauss_extent']) < 0:
            raise RuntimeError('"blur_gauss_extent" must be >= 0')
        if float(self.header['blur_spacing']) < 0:
            raise RuntimeError('"blur_spacing" must be >= 0')
        if float(self.header['blur_gauss_min']) < 0:
            raise RuntimeError('"blur_gauss_min" must be >= 0')

        self.header['blur_core_extent'] = float(self.header['blur_core_extent'])
        self.header['blur_gauss_extent'] = float(self.header['blur_gauss_extent'])
        self.header['blur_spacing'] = float(self.header['blur_spacing'])
        self.header['blur_gauss_min'] = float(self.header['blur_gauss_min'])

        if 'streamline_representation' not in self.header:
            self.header['streamline_representation'] = 'polyline'  # default value
            self.header['datatype'] = 'Float16'
            # TODO: Add 'control points' value in the near future
        if self.header['streamline_representation'] not in ['polyline']:
            raise RuntimeError('Problem parsing the header; field "streamline_representation" is not a valid value')

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if 'count' not in self.header:
            raise RuntimeError('Problem parsing the header; field "count" not found')
        if type(self.header['count']) == list:
            raise RuntimeError('Problem parsing the header; field "count" has multiple values')

        # TODO: is needed?
        # check if datatype is 'Float32LE'
        # if 'datatype' not in self.header:
        #     raise RuntimeError( 'Problem parsing the header; field "datatype" not found' )
        # if type(self.header['datatype'])==list:
        #     raise RuntimeError( 'Problem parsing the header; field "datatype" has multiple values' )
        # if self.header['datatype']!='Float32LE':
        #     raise RuntimeError( 'Unable to process file, as datatype "Float32LE" is not yet handled' )

        # move file pointer to beginning of binary data
        if 'file' not in self.header:
            raise RuntimeError('Problem parsing the header; field "file" not found')
        if type(self.header['file']) == list:
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
        cdef int offset = 25  # accounts for 'mrtrix tracks\n' and 'END\n'

        if header is None or type(header) != dict:
            raise RuntimeError('Provided header is empty or invalid')

        # check if the 'count' field is present TODO: fix this, allow working even without it
        if 'count' not in header:
            raise RuntimeError('Problem parsing the header; field "count" not found')
        if type(header['count']) == list:
            raise RuntimeError('Problem parsing the header; field "count" has multiple values')

        fseek(self.fp, 0, SEEK_SET)
        line = b'mrtrix track scalars\n'
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

    cpdef close(self, bint write_eof=True, int count=-1):
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

        if self.is_open == False:
            return

        if self.mode != 'r':
            # write end-of-file marker
            if write_eof:
                fwrite(&inf, 4, 1, self.fp)

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
