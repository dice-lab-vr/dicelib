#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

cdef float [:,::1] apply_affine(float [:,::1] end_pts, float [::1,:] M, float [:] abc, float [:,::1] end_pts_trans) noexcept nogil

cdef compute_grid( float thr, float[:] vox_dim ) 

cdef int[:] streamline_assignment( float [:] start_pt_grid, int[:] start_vox, float [:] end_pt_grid, int[:] end_vox, int [:] roi_ret, float [:,::1] mat, float [:,::1] grid, int[:,:,::1] gm_v, float thr, int[:] count_neighbours) noexcept nogil

