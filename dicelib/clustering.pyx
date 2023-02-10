#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

"""Functions to perform clustering of tractograms"""

import cython
import os
import numpy as np
cimport numpy as np
import nibabel as nib
from libc.math cimport sqrt, pow
from dipy.tracking.streamline import set_number_of_points as stp
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as tdp
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

# cdef float[:] tot_lenght(float[:,:] fib_in) :
#     cdef float[:] length = np.zeros(fib_in.shape[0], dtype=np.float32)
#     cdef size_t i = 0

#     length[0] = 0.0
#     for i in xrange(1,fib_in.shape[0]):
#         length[i] = length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 )
#     return length

cdef float[:] tot_lenght(float* ptr, int n) :
    cdef float[:] length_l = np.zeros(n, dtype=np.float32)
    cdef size_t i = 0
    
    # length[0] = 0.0
    cdef float* ptr_end = ptr+n*3-3
    length_l[0] = 0.0
    i += 1
    while ptr<ptr_end:
        length_l[i] = length_l[i-1] + sqrt( (ptr[3]-ptr[0])**2 + (ptr[4]-ptr[1])**2 + (ptr[5]-ptr[2])**2 )
        ptr += 3
        i+=1
    return length_l


# cdef float[:] tot_lenght_test(float[:,:] fib_in) :
#     cdef float[:] length = np.zeros(fib_in.shape[0], dtype=np.float32)
#     cdef size_t i = 0

#     length[0] = 0.0
#     for i in xrange(1,fib_in.shape[0]):
#         length[i] = sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 )
#         print(length[i])
#     return length


cdef float[:,:] set_number_of_points(float[:,:] fib_in, int nb_pts, float[:,:] resampled_fib) :
    cdef float[:] start = fib_in[0]
    cdef int nb_pts_in = fib_in.shape[0]
    cdef float[:] end = fib_in[nb_pts_in-1]
    cdef float [:] vers = np.zeros(3, dtype=np.float32)
    cdef size_t i = 0
    cdef size_t j = 0
    cdef float sum_step = 0
    cdef float[:] lenghts = tot_lenght(&fib_in[0,0], fib_in.shape[0])
    cdef float step_size = lenghts[nb_pts_in-1]/(nb_pts-1)
    cdef float sum_len = 0

    # for i in xrange(1, lenghts.shape[0]-1):
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    while sum_step < lenghts[nb_pts_in-1]:
        if i>10:break
        # if i>0:
            # print((fib_in[i-1][0], fib_in[i][0]))
            # print("")
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

cdef set_number_of_points_test(float[:,:]fib_in, int fib_in_shape, int nb_pts, float* ptr_resampled_fib) :
    cdef float [:] vers = np.zeros(3, dtype=np.float32)
    cdef size_t i = 0
    cdef float* ptr_fib_in = &fib_in[0,0]
    cdef float* end = ptr_fib_in+fib_in_shape*3-3
    cdef float sum_step = 0
    cdef float[:] lenghts = tot_lenght(ptr_fib_in, fib_in_shape)
    cdef float step_size = lenghts[fib_in_shape-1]/(nb_pts-1)
    cdef float sum_len = 0
    cdef float* ptr_fib_in_prev = ptr_fib_in

    # print((ptr_fib_in_prev[0]))
    # print((ptr_fib_in[0]))
    # print(f"last point:{fib_in[nb_pts_in-1][0], fib_in[nb_pts_in-1][1], fib_in[nb_pts_in-1][2]}")

    # for i in xrange(1, lenghts.shape[0]-1):
    ptr_resampled_fib[0] = ptr_fib_in[0]
    ptr_resampled_fib[1] = ptr_fib_in[1]
    ptr_resampled_fib[2] = ptr_fib_in[2]
    # ptr_resampled_fib += 3
    # ptr_fib_in += 3
    while sum_step < lenghts[nb_pts-1]:

        if sum_step == lenghts[i]:
            ptr_resampled_fib[0] = ptr_fib_in[0] 
            ptr_resampled_fib[1] = ptr_fib_in[1]
            ptr_resampled_fib[2] = ptr_fib_in[2]
            ptr_resampled_fib += 3
            # j += 1
            sum_step += step_size
        elif sum_step < lenghts[i]:
            ratio = 1 - ((lenghts[i]- sum_step)/(lenghts[i]-lenghts[i-1]))
            vers[0] = ptr_fib_in[0] - ptr_fib_in_prev[0]
            vers[1] = ptr_fib_in[1] - ptr_fib_in_prev[1]
            vers[2] = ptr_fib_in[2] - ptr_fib_in_prev[2]
            ptr_resampled_fib[0] = ptr_fib_in_prev[0] + ratio * vers[0]
            ptr_resampled_fib[1] = ptr_fib_in_prev[1] + ratio * vers[1]
            ptr_resampled_fib[2] = ptr_fib_in_prev[2] + ratio * vers[2]
            ptr_resampled_fib += 3
            # j += 1
            sum_step += step_size
        else:
            ptr_fib_in_prev = ptr_fib_in
            ptr_fib_in += 3
            i+=1

    ptr_resampled_fib[0] = end[0]
    ptr_resampled_fib[1] = end[1]
    ptr_resampled_fib[2] = end[2]


cdef (int, int) compute_dist(float* fib_in, float* target, int new_c, int thr,
                            float d1_x, float d1_y, float d1_z, float d2_x, float d2_y, float d2_z, float d3_x, float d3_y, floatd3_z,
                            float dt, float dm1_d, float dm1_i, float dm2, float dm3, int num_c, int num_pt) nogil:
    """Compute the distance between a fiber and a set of centroids"""
    # cdef float d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z
    # cdef float dt, dm1_d, dm1_i, dm2, dm3
    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef float maxdist_pt_d_test_1 = 0
    cdef float maxdist_pt_d_test_2 = 0
    cdef float maxdist_fib = 10000000000
    cdef int  i = 0
    cdef int  j = 0
    cdef int fib_idx = 0
    cdef int idx_ret = 0
    cdef int flipped = 0
    cdef int i_c = num_pt*3-3
    cdef float* start = fib_in
    # cdef float* ptr_fib_in_prev = &fib_in[0,0]

    for i in xrange(num_c):
        # i_c += new_c*num_pt*3-3
        # with gil:
        #     print(target[0], target[1], target[2])
        #     print(fib_in[0], fib_in[1], fib_in[2])
        #     print(fib_in[i_c], fib_in[i_c+1], fib_in[i_c+2])
        d1_x = pow(target[0] - fib_in[0],2)
        d1_y = pow(target[1] - fib_in[1],2)
        d1_z = pow(target[2] - fib_in[2],2)

        maxdist_pt_d = sqrt(d1_x + d1_y + d1_z)

        d1_x = pow(target[0] - fib_in[i_c],2)
        d1_y = pow(target[1] - fib_in[i_c+1],2)
        d1_z = pow(target[2] - fib_in[i_c+2],2)

        maxdist_pt_i = sqrt(d1_x + d1_y + d1_z)

        if maxdist_pt_d < maxdist_pt_i:
            flipped = 0
            # maxdist_pt_d = 0

            for j in xrange(1,num_pt):
                target += 3
                fib_in += 3
                d1_x = pow(target[0] - fib_in[0],2)
                d1_y = pow(target[1] - fib_in[1],2)
                d1_z = pow(target[2] - fib_in[2],2)

                maxdist_pt_d +=  sqrt(d1_x + d1_y + d1_z )

            fib_in = start

            if maxdist_pt_d < maxdist_fib:
                maxdist_fib = maxdist_pt_d
                idx_ret = i
        else:
            flipped = 1
            # maxdist_pt_i = 0
            for j in xrange(1,num_pt):
                i_c -= 3
                target += 3
                d1_x = pow(target[0] - fib_in[i_c],2)
                d1_y = pow(target[1] - fib_in[i_c+1],2)
                d1_z = pow(target[2] - fib_in[i_c+2],2)
                
                maxdist_pt_i += sqrt(d1_x + d1_y + d1_z)
            i_c = num_pt*3-3

            if maxdist_pt_i < maxdist_fib:
                maxdist_fib = maxdist_pt_i
                idx_ret = i
        target += 3

    # with gil:print( f"dist: {maxdist_fib/num_pt}\n")
    if maxdist_fib/num_pt < thr:
        return (idx_ret, flipped)

    return (num_c, flipped)


cpdef cluster(filename_in, filename_out=None, filename_reference=None, threshold=10, n_pts=10, replace_centroids=False,
             force=False, verbose=False):
    """ Cluster streamlines in a tractogram.
    TODO: DOCUMENTATION
    """
    print(f"\n\nQB v2.0 clustering thr: {threshold}, pts: {n_pts}")
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
    cdef float[:,::1] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:,::1] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
    s0 = next(tractogram_gen.streamlines)
    set_number_of_points_test(s0, s0.shape[0], nb_pts, &resampled_fib[0,0])

    # cdef float [:,::] s0 = np.array(stp(next(tractogram_gen.streamlines), nb_pts), dtype=np.float32)
    # for ii in range(s0.shape[0]):
    #         print(f"{s0[ii][0]},{s0[ii][1]},{s0[ii][2]}")
    # s0 = np.array(stp(next(tractogram_gen.streamlines), nb_pts), dtype=np.float32)
    # for ii in range(s0.shape[0]):
    #     print(f"\n{s0[ii][0]},{s0[ii][1]},{s0[ii][2]}")
    # return
    cdef float [:,::1] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,::1] streamline_in = np.zeros((nb_pts, 3), dtype=np.float32)
    cdef int[:] c_w = np.ones(n_streamlines, dtype=np.int32)
    cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
    cdef float [:] new_p_centr = np.zeros(3, dtype=np.float32)
    cdef float* pt_stream_in_start = &resampled_fib[0,0]
    cdef float* pt_stream_in_end = pt_stream_in_start + nb_pts*3-3
    cdef size_t  i, j = 0
    cdef int thr = threshold
    cdef int t = 0
    cdef int new_c = 1
    cdef int flipped = 0
    cdef int weight_centr = 0

    cdef float d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z
    cdef float dt, dm1_d, dm1_i, dm2, dm3

    cdef float* ptr_fib_in


    set_centroids[0] = resampled_fib
    clust_idx = np.zeros(n_streamlines, dtype=np.int32)
    t1 = time.time()

    for i, s in enumerate(tractogram_gen.streamlines):
        print(f"i:{i}, # clusters:{new_c}", end="\r")
        set_number_of_points_test(s, s.shape[0], nb_pts, &resampled_fib[0,0])
        pt_stream_in_start = &resampled_fib[0,0]
        pt_stream_in_end = pt_stream_in_start + nb_pts*3-3
        # for ii in range(resampled_fib.shape[0]):
        #     print(f"resampled {resampled_fib[ii][0]}, {resampled_fib[ii][1]}, {resampled_fib[ii][2]}")
        # print("")
        # streamline_in = stp(s, nb_pts)

        t, flipped = compute_dist(&resampled_fib[0,0], &set_centroids[0,0,0], new_c, thr, d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, d3_x, d3_y, d3_z,
                                  dt, dm1_d, dm1_i, dm2, dm3, set_centroids[:new_c].shape[0], nb_pts)
        clust_idx[i]= t
        weight_centr = c_w[t]
        if t < new_c:
            for p in xrange(nb_pts):
                pt_centr = set_centroids[t][p]

                if flipped:
                    new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in_start[0])/(weight_centr+1)
                    new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in_start[1])/(weight_centr+1)
                    new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in_start[2])/(weight_centr+1)

                    pt_stream_in_start += 3
                else:
                    new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in_end[0])/(weight_centr+1)
                    new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in_end[1])/(weight_centr+1)
                    new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in_end[2])/(weight_centr+1)

                    pt_stream_in_end -= 3

                new_centroid[p] = new_p_centr
                c_w[t] += 1
        else:
            for p in xrange(nb_pts):
                new_p_centr[0] = pt_stream_in_start[0]
                new_p_centr[1] = pt_stream_in_start[1]
                new_p_centr[2] = pt_stream_in_start[2]

                pt_stream_in_start += 3
                new_centroid[p] = new_p_centr
            new_c += 1

        set_centroids[t] = new_centroid

    print(f"time required: {np.round((time.time()-t1)/60, 3)} minutes")
    print(f"total_number of streamlines: {len(clust_idx)}")
    print(f"number of clusters {len(np.unique(clust_idx))}")

    return clust_idx

# cpdef run_cluster_parallel(filename_in, filename_out=None, filename_reference=None, threshold=10, n_pts=10, replace_centroids=False,
#              force=False, verbose=False):

#     print(f"\n\nQB v2.0 clustering thr: {threshold}, pts: {n_pts}")
#     ui.set_verbose( 2 if verbose else 1 )
#     if not os.path.isfile(filename_in):
#         ui.ERROR( f'File "{filename_in}" not found' )
#     if os.path.isfile(filename_out) and not force:
#         ui.ERROR("Output tractogram already exists, use -f to overwrite")
    
#     if filename_reference:
#         if not os.path.isfile(filename_reference):
#             ui.ERROR( f'File "{filename_reference}" not found' )
#         ui.INFO( f'Input tractogram: "{filename_in}"' )

#     if np.isscalar( threshold ) :
#         threshold = threshold

#     tractogram_gen = nib.streamlines.load(filename_in, lazy_load=True)
#     n_streamlines = int(tractogram_gen.header["count"])
#     ui.INFO( f'  - {n_streamlines} streamlines found' )

#     cdef int nb_pts = n_pts
#     cdef float[:,::] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
#     cdef float[:,:,::] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
#     cdef float [:,::] s0 = set_number_of_points(next(tractogram_gen.streamlines), nb_pts, resampled_fib)
#     cdef float [:,::] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)

#     set_centroids[0] = s0
#     # print(n_streamlines.shape)

#     clust_idx = np.zeros(n_streamlines, dtype=np.int32)
#     cdef size_t  i, j = 0
#     cdef int thr = threshold
#     cdef int t = 0
#     cdef float[:,::] streamline_in = np.zeros((nb_pts, 3), dtype=np.float32)
#     cdef int[:] c_w = np.ones(n_streamlines, dtype=np.int32)
#     cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
#     cdef float[:] pt_stream_in = np.zeros(3, dtype=np.float32)
#     cdef int new_c = 1
#     cdef float [:] new_p_centr = np.zeros(3, dtype=np.float32)
#     count_centr = 0
#     clust_idx[0] = 1
#     cdef int flipped = 0
#     cdef int N_THREAD = 1
#     cdef int MAX_THREAD = 2
#     chunks = [(0,1)]
#     cdef int weight_centr = 0
#     t1 = time.time()
#     executor = tdp(max_workers=MAX_THREAD)
    
#     cdef float[:,:] params_dist = np.zeros((MAX_THREAD,14), dtype=np.float32)
    
    

#     if MAX_THREAD>0:
#         print("\n\n RUNNING QB v2.0 multi thread")

#     for i, s in enumerate(tractogram_gen.streamlines):
#         print(f"i:{i}, # clusters:{new_c}", end="\r")
#         streamline_in = set_number_of_points(s, nb_pts, resampled_fib)

#         future = [executor.submit(compute_dist, streamline_in, set_centroids[i_c:c], thr,params_dist[i_p]) for i_c,c in chunks]
#         res_paral = [f.result() for f in future]
#         valid_idx = [r for r in res_paral if len(r)>1]

#         if len(valid_idx)>0:
#             i_t = np.argmin([k[1] for k in valid_idx])
#             t = valid_idx[i_t][0]
#             flipped = valid_idx[i_t][2]
#         else:
#             t = new_c
#         clust_idx[i]= t
#         weight_centr = c_w[t]
#         if t < new_c:

#             for p in xrange(nb_pts):
#                 pt_centr = set_centroids[t][p]
                
#                 if flipped:
#                     pt_stream_in = streamline_in[nb_pts-p-1]
#                     # print((t, flipped))
#                     new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
#                     new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
#                     new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
#                 else:
#                     pt_stream_in = streamline_in[p]
#                     new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
#                     new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
#                     new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
#                 new_centroid[p] = new_p_centr
#                 c_w[t] += 1

#         else:
#             new_centroid = streamline_in
#             new_c += 1
#             if N_THREAD < MAX_THREAD:
#                 N_THREAD += 1
#             c = new_c // N_THREAD
#             chunks = []
#             for i_d, j_d in zip(range(0, new_c-1, c), range(c, new_c+1, c)):
#                 chunks.append((i_d, j_d))
#             if chunks[len(chunks)-1][1] != new_c:
#                 chunks[len(chunks)-1] = (chunks[len(chunks)-1][0], new_c)

#         set_centroids[t] = new_centroid


#     print(f"time required: {np.round((time.time()-t1)/60, 3)} minutes")
#     print(f"number of clusters {len(np.unique(clust_idx))}")
    

#     return clust_idx