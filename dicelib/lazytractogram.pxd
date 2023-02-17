#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3


from libc.stdio cimport fopen, fclose, FILE#, fseek, SEEK_END, SEEK_SET

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
    cdef void _write_streamline( self ) nogil