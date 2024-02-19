#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

cdef float [:] apply_affine_1pt(float [:] orig_pt, double[::1,:] M, double[:] abc, float [:] moved_pt)