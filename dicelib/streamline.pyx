#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False, nonecheck=False, cdivision=True, initializedcheck=False, binding=False
import cython
from bisect import bisect_right
from libc.math cimport sqrt, floor
import numpy as np
cimport numpy as np




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



cpdef double [:,::1] space_tovox(streamline, header,curr_space = None ):
    """Method to change space reference of streamlines (.tck)
    Note that if curr_space is None, space is interpreted as RASmm 

    Allowed spaces tranformation: 
    voxmm --> vox 
    rasmm --> vox

    Parameters:

    -----------

    streamline : Numpy array Nx3
        Data of the streamline, coordinates 
    header : NiftiHeader 
        header of the image
    curr space : string 
        coordinates space of streamline to transform
    """
    streamline = np.asarray(streamline)
    voxsize = np.asarray(header["pixdim"][1:4]) #resolution
    affine = np.asarray([header["srow_x"],header["srow_y"],header["srow_z"],[0,0,0,1]]) #affine retrieved from the header
    inverse = np.linalg.inv(affine) #inverse of affine
    small = inverse[:-1,:-1].T 
    val = inverse[:-1,-1]
    # if curr_space == "voxmm":
    #     streamline /= voxsize
    # elif curr_space == "rasmm": 
    #     streamline = np.matmul(streamline,inverse[:-1,:-1].T) + inverse[:-1,-1] #same as nibabel.affines.apply_affine() 
    #     streamline += voxsize/2 #to point center of the voxel 
    #     streamline = np.floor(streamline) #cast 


    if not streamline.flags['C_CONTIGUOUS']:
        streamline = np.ascontiguousarray(streamline)
    if not small.flags['C_CONTIGUOUS']:
        small = np.ascontiguousarray(small)
    if not val.flags['C_CONTIGUOUS']:
        val = np.ascontiguousarray(val)
    if not small.flags['C_CONTIGUOUS']:
        small = np.ascontiguousarray(small)

    cdef double [:,::1] streamline_view = np.double(streamline)
    cdef double [:,::1] small_view = small
    cdef float  [::1] voxsize_view = voxsize
    cdef double [::1] val_view = val  
    cdef double somma = 0.0
    cdef size_t ii, yy

    if curr_space == "voxmm":
        for ii in range(streamline_view.shape[0]):
            for yy in range(streamline_view.shape[1]):
                streamline_view[ii][yy] = streamline_view[ii][yy]/voxsize_view[yy]
    else :     #rasmm 
        for ii in range(streamline_view.shape[0]):
            for yy in range(streamline_view.shape[1]):
                somma = ((streamline_view[ii,0]*small_view[0,yy] + streamline_view[ii,1]*small_view[1,yy] + streamline_view[ii,2]*small_view[2,yy]) + val_view[yy]) 
                somma += (voxsize_view[yy]/2)
                streamline_view[ii][yy] = floor(somma)
    
    return streamline_view


cdef float [:,::1] apply_affine(float [:,::1] end_pts, float [::1,:] M,
                                float [:] abc, float [:,::1] end_pts_trans) noexcept nogil:

    # N.B. use this function only to move from RASmm to VOX, not the inverse (because of +0.5)
    end_pts_trans[0][0] = ((end_pts[0][0]*M[0,0] + end_pts[0][1]*M[1,0] + end_pts[0][2]*M[2,0]) + abc[0]) +0.5
    end_pts_trans[0][1] = ((end_pts[0][0]*M[0,1] + end_pts[0][1]*M[1,1] + end_pts[0][2]*M[2,1]) + abc[1]) +0.5
    end_pts_trans[0][2] = ((end_pts[0][0]*M[0,2] + end_pts[0][1]*M[1,2] + end_pts[0][2]*M[2,2]) + abc[2]) +0.5
    end_pts_trans[1][0] = ((end_pts[1][0]*M[0,0] + end_pts[1][1]*M[1,0] + end_pts[1][2]*M[2,0]) + abc[0]) +0.5
    end_pts_trans[1][1] = ((end_pts[1][0]*M[0,1] + end_pts[1][1]*M[1,1] + end_pts[1][2]*M[2,1]) + abc[1]) +0.5
    end_pts_trans[1][2] = ((end_pts[1][0]*M[0,2] + end_pts[1][1]*M[1,2] + end_pts[1][2]*M[2,2]) + abc[2]) +0.5


    return end_pts_trans

cdef float [:] apply_affine_1pt(float [:] orig_pt, double[::1,:] M, double[:] abc, float [:] moved_pt):
    moved_pt[0] = float((orig_pt[0]*M[0,0] + orig_pt[1]*M[1,0] + orig_pt[2]*M[2,0]) + abc[0]) + .5
    moved_pt[1] = float((orig_pt[0]*M[0,1] + orig_pt[1]*M[1,1] + orig_pt[2]*M[2,1]) + abc[1]) + .5
    moved_pt[2] = float((orig_pt[0]*M[0,2] + orig_pt[1]*M[1,2] + orig_pt[2]*M[2,2]) + abc[2]) + .5
    return moved_pt



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


cpdef apply_smoothing(fib_ptr, n_pts_in, segment_len=None, n_pts_final=None, epsilon = 0.3, alpha = 0.5, n_pts_tmp = 50):
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
    cdef int n_pts_out
    if segment_len!=None:
        n_pts_out = int(fib_len / segment_len)
    if n_pts_final!=None:
        n_pts_out = n_pts_final

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
    cdef float[:] lengths = np.zeros(nb_pts_in, dtype=np.float32)
    cdef float[:,::1] fib_in = np.ascontiguousarray(streamline, dtype=np.float32)
    
    resample_len(fib_in, &lengths[0])

    cdef float step_size = lengths[nb_pts_in-1]/(nb_pts-1)
    cdef float sum_len = 0
    cdef float ratio = 0

    # for i in xrange(1, lengths.shape[0]-1):
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    while sum_step < lengths[nb_pts_in-1]:
        if sum_step == lengths[i]:
            resampled_fib[j][0] = fib_in[i][0] 
            resampled_fib[j][1] = fib_in[i][1]
            resampled_fib[j][2] = fib_in[i][2]
            j += 1
            sum_step += step_size
        elif sum_step < lengths[i]:
            ratio = 1 - ((lengths[i]- sum_step)/(lengths[i]-lengths[i-1]))
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


cpdef sampling(float [:,::1] streamline_view, float [:,:,::1] img_view, int npoints, float [:,:,::1] mask_view, option=None):
    """Compute the length of a streamline.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data, coordinates 
    img : numpy array 
        data of the image 
    npoints : int  
        points of the streamline 
    Returns
    -------
    value : numpy array of dim (npoint,)
        values that correspond to coordinates of streamline in the image space 
         
    """

    value = np.empty([npoints,], dtype= float) 
    opt_value = 0
    cdef size_t ii

    for ii in range(npoints):
        vox_coords = np.array([int(streamline_view[ii,0]), int(streamline_view[ii,1]), int(streamline_view[ii,2])])
        if mask_view[<int>vox_coords[0], <int>vox_coords[1], <int>vox_coords[2]] == 0:
            value[ii] = np.nan
        else: 
            value[ii] = img_view[<int>streamline_view[ii,0],<int>streamline_view[ii,1],<int>streamline_view[ii,2]] #cast int values
         
        
    if option == "mean":
        opt_value = np.nanmean(value)
        return opt_value
    elif option == "median":
        opt_value = np.nanmedian(value)
        return opt_value
    elif option == "min":
        opt_value = value.min()
        return opt_value
    elif option == "max":
        opt_value = value.max()
        return opt_value
    else: #none case
        return value


cpdef void set_number_of_points(float[:,::1] fib_in, int nb_pts, float[:,::1] resampled_fib, float[::1] vers, float[::1] lengths):# noexcept nogil:
    cdef int nb_pts_in = fib_in.shape[0]
    cdef size_t i = 0
    cdef size_t j = 0
    cdef float sum_step = 0
    tot_lenght(fib_in, lengths)

    cdef float step_size = lengths[nb_pts_in-1]/(nb_pts-1)
    cdef float ratio = 0

    # for i in xrange(1, lengths.shape[0]-1):
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    while sum_step < lengths[nb_pts_in-1]:
        if sum_step == lengths[i]:
            resampled_fib[j][0] = fib_in[i][0] 
            resampled_fib[j][1] = fib_in[i][1]
            resampled_fib[j][2] = fib_in[i][2]
            j += 1
            sum_step += step_size
        elif sum_step < lengths[i]:
            ratio = 1 - ((lengths[i]- sum_step)/(lengths[i]-lengths[i-1]))
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


cdef void tot_lenght(float[:,::1] fib_in, float[::1] length):# noexcept nogil:
    cdef size_t i = 0

    length[0] = 0.0
    for i in xrange(1,fib_in.shape[0]):
        length[i] = <float>(length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 ))


cdef float[:] compute_tangent(float[:,:] points, float[:] grid):
    cdef float[:] x0 = points[0]
    cdef float[:] x1 = points[1]
    cdef float[:] x2 = points[2]   
    cdef float t0 = grid[0]
    cdef float t1 = grid[1]
    cdef float t2 = grid[2]
    cdef float delta0 = t2 -t1
    cdef float delta1 = t1 - t0
    cdef float[:] v0 = np.empty((3,), dtype=np.float32)
    cdef float[:] v1 = np.empty((3,), dtype=np.float32)
    cdef float[:] tangent = np.empty((3,), dtype=np.float32)
    cdef size_t i = 0

    for i in range(3):
        v0[i] = (x2[i] - x1[i]) / delta0
        v1[i] = (x1[i] - x0[i]) / delta1
    for i in range(3):
        tangent[i] = (delta0 * v1[i] + delta1 * v0[i]) / (delta0 + delta1)
        
    return tangent


cdef float[:, :] CatmullRom_smooth(float[:, :] vertices, float[:, :] matrix, float alpha=0.5, int num_pts=10):
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0
    cdef int idx_temp = 0
    # cdef float[:] x0 = np.empty((3,), dtype=np.float32)
    # cdef float[:] x1 = np.empty((3,), dtype=np.float32)
    # cdef float[:] v0 = np.empty((3,), dtype=np.float32)
    # cdef float[:] v1 = np.empty((3,), dtype=np.float32)
    cdef float t0 = 0
    cdef float t1 = 0

    cdef float[:] t = np.empty((num_pts,), dtype=np.float32)
    cdef float[:, :] tangent = np.empty((vertices.shape[0]-2, 3), dtype=np.float32)
    cdef float[:,:] tangents = np.empty((2*tangent.shape[0]+2, 3), dtype=np.float32)
    cdef float[:, :, :] segments = np.empty((vertices.shape[0]-1, 4, 3), dtype=np.float32)
    cdef float[:] grid = np.empty(vertices.shape[0], dtype=np.float32)
    cdef float[:,:] prod = np.empty((4, 3), dtype=np.float32)

    grid = check_grid(grid, alpha, vertices)
    cdef float[:, :] smoothed = np.empty((num_pts, 3), dtype=np.float32)

    for i in range(vertices.shape[0]):
        # compute tangent over triplets of vertices and grid points
        if i < vertices.shape[0]-2:
            tangent[i] = compute_tangent(vertices[i:i+3], grid[i:i+3])

    # fill tangents array by duplicating each value of tangent starting from the second
    for i in range(tangent.shape[0]):
        tangents[2*i+1] = tangent[i]
        tangents[2*i+2] = tangent[i]
    

    # Calculate tangent for "natural" end condition
    x0, x1 = vertices[0], vertices[1]
    t0, t1 = grid[0], grid[1]
    delta = t1 - t0

    for i in range(3):
        tangents[0][i] = 3 * (x1[i] - x0[i]) / (2*delta) - tangent[0][i] / 2

    x0, x1 = vertices[vertices.shape[0]-2], vertices[vertices.shape[0]-1]
    t0, t1 = grid[grid.shape[0]-2], grid[grid.shape[0]-1]
    delta = t1 - t0
    for i in range(3):
        tangents[tangents.shape[0]-1][i] = 3 * (x1[i] - x0[i]) / (2*delta) - tangent[tangent.shape[0]-1][i] / 2

    for i in range(vertices.shape[0]-1):
        x0 = np.asarray(vertices[i])
        x1 = np.asarray(vertices[i+1])
        v0 = np.asarray(tangents[2*i])
        v1 = np.asarray(tangents[2*i+1])
        t0 = grid[i]
        t1 = grid[i+1]
        
        # for j in range(4):
        #     for k in range(3):
        #         prod[j][k] = matrix[j][0] * x0[k] + matrix[j][1] * x1[k] + matrix[j][2] * v0[k] + matrix[j][3] * v1[k]
        prod = matrix @ np.array([x0, x1, (t1 - t0) * v0, (t1 - t0) * v1])      
        for j in range(4):
            for k in range(3):
                segments[i][j][k] = prod[j][k]

    t = np.linspace(0, np.array(grid).max(), num_pts).astype(np.float32)
    for i in range(t.shape[0]):
        if t[i] < grid[grid.shape[0]-1]:
            idx_temp = bisect_right(grid, t[i]) - 1
        else:
            idx_temp = len(grid) - 2

        t0, t1 = grid[idx_temp:idx_temp+2]
        tt = (t[i] - t0) / (t1 - t0)
        coefficients = segments[idx_temp]
        powers = np.arange(len(coefficients))[::-1]
        new_val = tt**powers @ coefficients
        for j in range(3):
            smoothed[i][j] = new_val[j]
    return smoothed


cdef float[:] check_grid(float[:] grid, float alpha, float[:, :] vertices):
    cdef size_t i = 1
    cdef size_t ii = 0
    cdef size_t jj = 0
    cdef float diff = 0
    cdef float[:] x0 = np.empty((3,), dtype=np.float32)
    cdef float[:] x1 = np.empty((3,), dtype=np.float32)

    if alpha == 0:
        # NB: This is the same as alpha=0, except the type is int
        for jj in range(vertices.shape[0]):
            grid[jj] = jj

    grid[0] = 0
    for ii in range(vertices.shape[0]-1):
        for jj in range(3):
            x0[jj] = vertices[ii][jj]
            x1[jj] = vertices[ii+1][jj]
        
        # rewrite diff to avoid numpy overhead
        diff = np.sqrt((x1[0] - x0[0])**2 + (x1[1] - x0[1])**2 + (x1[2] - x0[2])**2)**alpha
        # x0 = np.asarray(vertices[ii])
        # x1 = np.asarray(vertices[ii+1])
        # diff = np.linalg.norm(x1 - x0)**alpha

        if diff == 0:
            raise ValueError(
                'Repeated vertices are not possible with alpha != 0')
        grid[i] = grid[i-1] + diff
        i += 1
    return grid


cpdef spline_smooth(vertices, alpha=0.5, num_pts=10):
    cdef float[:, :] matrix = np.array([ [2, -2, 1, 1],
                                          [-3, 3, -2, -1],
                                          [0, 0, 1, 0],
                                          [1, 0, 0, 0]]).astype(np.float32)
    vertices = np.asarray(vertices).astype(np.float32)
    smoothed = np.asarray(CatmullRom_smooth(vertices, matrix, alpha, num_pts))

    return smoothed


