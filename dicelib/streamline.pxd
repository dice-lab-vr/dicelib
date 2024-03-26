# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

cdef float [:,::1] apply_affine(float [:,::1] end_pts, float [::1,:] M, float [:] abc, float [:,::1] end_pts_trans) noexcept nogil

cdef float [:] apply_affine_1pt(float [:] orig_pt, double[::1,:] M, double[:] abc, float [:] moved_pt)