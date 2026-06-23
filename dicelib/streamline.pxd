# cython: boundscheck=False, wraparound=False, profile=False, language_level=3
cdef void apply_xform_to_point(float[:] in_P, double[:,::1] M, float[:] out_P) noexcept nogil
cdef void set_number_of_points(float[:,::1] in_streamline, int in_n_pts, float[:,::1] out_streamline, int out_n_pts, float[:] lengths) noexcept nogil