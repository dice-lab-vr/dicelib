#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt


cdef extern from "streamline.hpp":
    int smooth_c(
        float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len
    ) nogil


cpdef length( float [:,:] streamline, int n=0 ):
    """Compute the length of a streamline.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data
    n : int
        Writes first n points of the streamline. If n<=0 (default), writes all points

    Returns
    -------
    length : double
        Length of the streamline in mm
    """
    if n<0:
        n = streamline.shape[0]
    cdef float* ptr     = &streamline[0,0]
    cdef float* ptr_end = ptr+n*3-3
    cdef double length = 0.0
    while ptr<ptr_end:
        length += sqrt( (ptr[3]-ptr[0])**2 + (ptr[4]-ptr[1])**2 + (ptr[5]-ptr[2])**2 )
        ptr += 3
    return length


cpdef smooth( streamline, n_pts, control_point_ratio, segment_len ):
    """Wrapper for streamline smoothing.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data
    n_pts : int
        Number of points in the streamline
    control_point_ratio : float
        Ratio of control points w.r.t. the number of points of the input streamline
    segment_len : float
        Min length of the segments in mm

    Returns
    -------
    streamline_out : Nx3 numpy array
        The smoothed streamline data
    n : int
        Number of points in the smoothed streamline
    """

    cdef float [:,:] streamline_in = streamline
    cdef float [:,:] streamline_out = np.ascontiguousarray( np.zeros( (3*1000,1) ).astype(np.float32) )
    
    n = smooth_c( &streamline_in[0,0], n_pts, &streamline_out[0,0], control_point_ratio, segment_len )
    if n != 0 :
        streamline = np.reshape( streamline_out[:3*n].copy(), (n,3) )
    return streamline, n

