# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

cdef void apply_xform_to_point(float[:] in_P, double[:,::1] M, float[:] out_P) noexcept nogil