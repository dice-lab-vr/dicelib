# cython: boundscheck=False, wraparound=False, profile=False, language_level=3
from libc.stdio cimport FILE

cdef class Tsf:
    cdef readonly   str                             filename
    cdef readonly   str                             suffix
    cdef readonly   dict                            header
    cdef readonly   str                             mode
    cdef readonly   bint                            is_open
    cdef            FILE*                           fp

    cpdef _read_header( self )
    cpdef _write_header( self, header )
    cpdef write_scalar( self, scalars, pts )
    cpdef read_scalar( self )
    cpdef close( self, bint write_eof=*, int count=* )