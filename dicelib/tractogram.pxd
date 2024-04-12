# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

from libc.stdio cimport fclose, FILE, fopen

cdef class LazyTractogram:
    cdef readonly   str                             filename
    cdef readonly   str                             suffix
    cdef readonly   dict                            header
    cdef readonly   str                             mode
    cdef readonly   bint                            is_open
    cdef readonly   float[:,::1]                      streamline
    cdef readonly   unsigned int                    n_pts
    cdef            int                             max_points
    cdef            FILE*                           fp
    cdef            float*                          buffer
    cdef            float*                          buffer_ptr
    cdef            float*                          buffer_end

    cdef int _read_streamline( self ) nogil
    cpdef int read_streamline( self )
    cdef void _write_streamline( self, float [:,:] streamline, int n=* ) nogil
    cpdef write_streamline( self, float [:,:] streamline, int n=* )
    cpdef close( self, bint write_eof=*, int count=* )
    cpdef _read_header( self )
    cpdef _write_header( self, header )
    cdef void _seek_origin( self, int header_param ) nogil
    cdef void move_to( self, int pts ) nogil