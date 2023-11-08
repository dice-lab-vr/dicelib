#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from dicelib.smoothing import spline_smooth


cdef extern from "streamline.hpp":
    int smooth_c(
        float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len
    ) nogil

cdef extern from "streamline.hpp":
    int rdp_red_c(
        float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float epsilon
    ) nogil

cdef extern from "streamline.hpp":
    int create_replicas_pt( 
        float* ptr_pts_in, double* ptr_pts_out, double* ptr_blur_rho, double* ptr_blur_angle, int n_replicas, float fiberShiftX, float fiberShiftY, float fiberShiftZ 
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


cpdef rdp_reduction( streamline, n_pts, epsilon ):
    """Wrapper for streamline point reduction.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data
    n_pts : int
        Number of points in the streamline
    epsilon : float
        Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points

    Returns
    -------
    streamline : Nx3 numpy array
        The smoothed streamline data
    n : int
        Number of points in the smoothed streamline
    """
    
    cdef float [:,:] streamline_in = streamline
    cdef float [:,:] streamline_out = np.ascontiguousarray( np.zeros( (3*1000,1) ).astype(np.float32) )
    
    n = rdp_red_c( &streamline_in[0,0], n_pts, &streamline_out[0,0], epsilon )
    if n != 0 :
        streamline = np.reshape( streamline_out[:3*n].copy(), (n,3) )

    return streamline, n


cpdef apply_smoothing(fib_ptr, n_pts_in, segment_len, epsilon = 0.3, alpha = 0.5, n_pts_tmp = 50):
    """Perform smoothing on one streamline.

    Parameters
    ----------
    fib_ptr : Nx3 numpy array
        The streamline data
    n_pts_in : int
        Number of points in the streamline
    segment_len : float
        Min length of the segments in mm
    epsilon : float
        Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline (default : 0.3)
    alpha : float
        Parameter defining the spline type: 0.5 = 'centripetal', 0.0 = 'uniform' or 1.0 = 'chordal' (default : 0.5).
    n_pts_temp : int
        Number of points used for the first sampling of the spline

    Returns
    -------
    resampled_fib : Nx3 numpy array
        The smoothed streamline data
    n_pts_out : int
        Number of points in the smoothed streamline
    """

    # reduce number of points
    fib_red_ptr, n_red = rdp_reduction(fib_ptr, n_pts_in, epsilon)

    cdef float [:,:] smoothed_fib
    cdef int n_pts_tot = 0
    # check number of points 
    if n_red==2: # no need to smooth
        smoothed_fib = fib_red_ptr
        n_pts_tot = n_red
    else:
        # get reduced streamline as np array
        # fib_reduced = np.asarray(fib_red_ptr, dtype=np.float32)[:n_red, :]
        # compute spline
        # smoothed_spline = splines.CatmullRom(fib_reduced, alpha=alpha)
        # sample spline
        # smoothed_fib_arr = smoothed_spline.evaluate(np.linspace(0, np.array(smoothed_spline.grid).max(), n_pts_tmp)).astype(np.float32)
        smoothed_fib = spline_smooth(fib_red_ptr, alpha, n_pts_tmp)
        # get spline as memory view
        # smoothed_fib = np.ascontiguousarray(smoothed_fib_arr)
        n_pts_tot = n_pts_tmp

    # compute streamline length
    cdef float fib_len = length( smoothed_fib, n_pts_tot )
    # compute number of final points
    cdef int n_pts_out = int(fib_len / segment_len)

    # resample smoothed streamline
    cdef float [:,:] resampled_fib = resample(smoothed_fib, n_pts_out)

    return resampled_fib, n_pts_out


cpdef resample (streamline, nb_pts) :
    cdef int nb_pts_in = streamline.shape[0]
    cdef resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef size_t i = 0
    cdef size_t j = 0
    cdef float sum_step = 0
    cdef float[:] vers = np.zeros(3, dtype=np.float32)
    cdef float[:] lenghts = np.zeros(nb_pts_in, dtype=np.float32)
    cdef float[:,::1] fib_in = np.ascontiguousarray(streamline, dtype=np.float32)
    
    resample_len(fib_in, &lenghts[0])

    cdef float step_size = lenghts[nb_pts_in-1]/(nb_pts-1)
    cdef float sum_len = 0
    cdef float ratio = 0

    # for i in xrange(1, lenghts.shape[0]-1):
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    while sum_step < lenghts[nb_pts_in-1]:
        if sum_step == lenghts[i]:
            resampled_fib[j][0] = fib_in[i][0] 
            resampled_fib[j][1] = fib_in[i][1]
            resampled_fib[j][2] = fib_in[i][2]
            j += 1
            sum_step += step_size
        elif sum_step < lenghts[i]:
            ratio = 1 - ((lenghts[i]- sum_step)/(lenghts[i]-lenghts[i-1]))
            vers[0] = fib_in[i][0] - fib_in[i-1][0]
            vers[1] = fib_in[i][1] - fib_in[i-1][1]
            vers[2] = fib_in[i][2] - fib_in[i-1][2]
            resampled_fib[j][0] = fib_in[i-1][0] + ratio * vers[0]
            resampled_fib[j][1] = fib_in[i-1][1] + ratio * vers[1]
            resampled_fib[j][2] = fib_in[i-1][2] + ratio * vers[2]
            j += 1
            sum_step += step_size
        else:
            i+=1
    resampled_fib[nb_pts-1][0] = fib_in[nb_pts_in-1][0]
    resampled_fib[nb_pts-1][1] = fib_in[nb_pts_in-1][1]
    resampled_fib[nb_pts-1][2] = fib_in[nb_pts_in-1][2]

    return resampled_fib


cdef void resample_len(float[:,::1] fib_in, float* length):
    cdef size_t i = 0

    length[0] = 0.0
    for i in xrange(1,fib_in.shape[0]):
        length[i] = <float>(length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 ))


cpdef float [:,::1] create_replicas( float [:,::1] in_pts, double [:] blurRho, double [:] blurAngle, int nReplicas, float fiber_shiftX, float fiber_shiftY, float fiber_shiftZ):
    """ Generate the replicas of an ending point given the grid coordinates.
    
    Parameters
    -----------------
    
    TODO

    """

    cdef double [:,:] pts_out = np.ascontiguousarray( np.zeros( (3*nReplicas,1) ).astype(np.float64) )
    n = create_replicas_pt( &in_pts[0,0], &pts_out[0,0], &blurRho[0], &blurAngle[0], nReplicas, fiber_shiftX, fiber_shiftY, fiber_shiftZ )
    if n != 0 :
        out_pts_m = np.reshape( pts_out[:3*n].copy(), (n,3) ).astype(np.float32)

    return out_pts_m
