# cython: boundscheck=False, wraparound=False, profile=False, language_level=3
from libc.stdio cimport FILE

cdef class TrackScalarFile:
    cdef readonly   str                             filename
    cdef readonly   str                             suffix
    cdef readonly   dict                            header
    cdef readonly   str                             mode
    cdef readonly   bint                            is_open
    cdef readonly   float[:]                        scalars
    cdef readonly   unsigned int                    n_pts
    cdef            int                             max_points
    cdef            FILE*                           fp

    cpdef int read_scalars( self ) nogil
    cpdef void write_scalars( self, float [:] scalars, int n=* ) nogil
    cpdef close( self, bint write_eof=*, int count=* )
    cpdef _read_header( self )
    cpdef _write_header( self, header )
