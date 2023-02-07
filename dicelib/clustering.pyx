#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

"""Functions to perform clustering of tractograms"""

import cython
import os
import numpy as np
cimport numpy as np
import nibabel as nib
from libc.math cimport sqrt
# from dipy.tracking.streamline import set_number_of_points
import time
from tqdm import tqdm
from . import ui


def get_streamlines_close_to_centroids( clusters, streamlines, n_pts ):
    """Return the streamlines closer to the centroids of each cluster.

    As first step, the streamlines of the input tractogram are resampled to n_pts points.
    """
    cdef float[:,:] resampled_fib = np.zeros((n_pts,3), dtype=np.float32)
    sample_streamlines = set_number_of_points(streamlines, n_pts, resampled_fib)

    centroids_out = []
    for cluster in clusters:
        minDis      = 1e10
        minDis_idx  = -1
        centroid_fw = cluster.centroid
        centroid_bw = cluster.centroid[::-1]
        for i in cluster.indices:
            d1 = np.linalg.norm( centroid_fw - sample_streamlines[i] )
            d2 = np.linalg.norm( centroid_bw - sample_streamlines[i] )
            if d1>d2:
                dm = d2
            else:
                dm = d1

            if dm < minDis:
                minDis = dm
                minDis_idx = i
        centroids_out.append( streamlines[minDis_idx] )

    return centroids_out

cdef float[:] tot_lenght(float[:,:] fib_in) :
    cdef float[:] length = np.zeros(fib_in.shape[0], dtype=np.float32)
    cdef size_t i = 0

    length[0] = 0.0
    for i in xrange(1,fib_in.shape[0]):
        length[i] = length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 )
    return length

cdef float[:,:] set_number_of_points(float[:,:] fib_in, int nb_pts, float[:,:] resampled_fib) :
    cdef float[:] start = fib_in[0]
    cdef int nb_pts_in = fib_in.shape[0]
    cdef float[:] end = fib_in[nb_pts_in-1]
    cdef float [:] vers = np.zeros(3, dtype=np.float32)
    cdef size_t i = 0
    cdef size_t j = 0
    cdef float sum_step = 0
    cdef float[:] lenghts = tot_lenght(fib_in)
    cdef float step_size = lenghts[nb_pts_in-1]/nb_pts
    cdef float sum_len = 0 

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


cdef (int, int) compute_dist(float[:,:] fib_in, float[:,:,:] target, int thr,
                            float d1_x, float d1_y, float d1_z, float d2_x, float d2_y, float d2_z, float d3_x, float d3_y, floatd3_z,
                            float dt, float dm1_d, float dm1_i, float dm2, float dm3, int num_c, int num_pt) nogil:
    """Compute the distance between a fiber and a set of centroids"""
    # cdef float d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z
    # cdef float dt, dm1_d, dm1_i, dm2, dm3
    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef float maxdist_fib = 10000000000
    cdef int  i = 0
    cdef int  j = 0
    cdef int fib_idx = 0
    cdef int idx_ret = 0
    cdef int flipped = 0
    # cdef int num_c = target.shape[0]
    # cdef int num_pt = target.shape[1]

    for i in xrange(num_c):
        maxdist_pt_d = 0
        maxdist_pt_i = 0

        for j in xrange(num_pt):
            d1_x = (target[i][j][0] - fib_in[j][0])**2
            d1_y = (target[i][j][1] - fib_in[j][1])**2
            d1_z = (target[i][j][2] - fib_in[j][2])**2

            maxdist_pt_d += sqrt(d1_x + d1_y + d1_z)

            d1_x = (target[i][j][0] - fib_in[num_pt-j-1][0])**2
            d1_y = (target[i][j][1] - fib_in[num_pt-j-1][1])**2
            d1_z = (target[i][j][2] - fib_in[num_pt-j-1][2])**2
            
            maxdist_pt_i += sqrt(d1_x + d1_y + d1_z)

        if maxdist_pt_d < maxdist_pt_i:
            maxdist_pt = maxdist_pt_d
            flipped = 0
        else:
            maxdist_pt = maxdist_pt_i
            flipped = 1
        
        if maxdist_pt < maxdist_fib:
            maxdist_fib = maxdist_pt
            idx_ret = i

    if maxdist_fib/num_pt < thr:
        return (idx_ret, flipped)

    return (num_c, flipped)#, -1, flipped)


cpdef cluster(filename_in, filename_out=None, filename_reference=None, threshold=10, n_pts=10, replace_centroids=False,
             force=False, verbose=False):
    """ Cluster streamlines in a tractogram.
    TODO: DOCUMENTATION
    """

    ui.set_verbose( 2 if verbose else 1 )
    if not os.path.isfile(filename_in):
        ui.ERROR( f'File "{filename_in}" not found' )
    if os.path.isfile(filename_out) and not force:
        ui.ERROR("Output tractogram already exists, use -f to overwrite")
    
    if filename_reference:
        if not os.path.isfile(filename_reference):
            ui.ERROR( f'File "{filename_reference}" not found' )
        ui.INFO( f'Input tractogram: "{filename_in}"' )

    if np.isscalar( threshold ) :
        threshold = threshold

    tractogram_gen = nib.streamlines.load(filename_in, lazy_load=True)
    n_streamlines = int(tractogram_gen.header["count"])
    ui.INFO( f'  - {n_streamlines} streamlines found' )

    cdef int nb_pts = n_pts
    cdef float[:,::] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:,::] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
    cdef float [:,::] s0 = np.array(set_number_of_points(next(tractogram_gen.streamlines), nb_pts, resampled_fib), dtype=np.float32)
    cdef float [:,::] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,::] streamline_in = np.zeros((nb_pts, 3), dtype=np.float32)
    cdef int[:] c_w = np.ones(n_streamlines, dtype=np.int32)
    cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_stream_in = np.zeros(3, dtype=np.float32)
    cdef float [:] new_p_centr = np.zeros(3, dtype=np.float32)
    cdef size_t  i, j = 0
    cdef int thr = threshold
    cdef int t = 0
    cdef int new_c = 1
    cdef int flipped = 0
    cdef int weight_centr = 0

    cdef float d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z
    cdef float dt, dm1_d, dm1_i, dm2, dm3


    set_centroids[0] = s0
    clust_idx = np.zeros(n_streamlines, dtype=np.int32)
    t1 = time.time()

    for i, s in enumerate(tractogram_gen.streamlines):
        print(f"i:{i}, # clusters:{new_c}", end="\r")
        streamline_in = set_number_of_points(s, nb_pts, resampled_fib)
        t, flipped = compute_dist(streamline_in, set_centroids[:new_c], thr, d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z,
                                  dt, dm1_d, dm1_i, dm2, dm3, set_centroids[:new_c].shape[0], nb_pts)
        clust_idx[i]= t
        weight_centr = c_w[t]
        if t < new_c:
            for p in xrange(nb_pts):
                pt_centr = set_centroids[t][p]
                
                if flipped:
                    pt_stream_in = streamline_in[nb_pts-p-1]
                    new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                    new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                    new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                else:
                    pt_stream_in = streamline_in[p]
                    new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                    new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                    new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                new_centroid[p] = new_p_centr
                c_w[t] += 1

        else:
            new_centroid = streamline_in
            new_c += 1

        set_centroids[t] = new_centroid

    print(f"time required: {np.round((time.time()-t1)/60, 3)} minutes")
    print(f"total_number of streamlines: {len(clust_idx)}")
    print(f"number of clusters {len(np.unique(clust_idx))}")

    return clust_idx