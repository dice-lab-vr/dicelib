#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

"""Functions to perform clustering of tractograms"""

import cython
import os
import numpy as np
cimport numpy as np
import nibabel as nib
from lazytractogram cimport LazyTractogram
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from geom_clustering import split_clusters
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as tdp
from . import ui


def get_streamlines_close_to_centroids( clusters, streamlines, n_pts ):
    """Return the streamlines closer to the centroids of each cluster.

    As first step, the streamlines of the input tractogram are resampled to n_pts points.
    """
    cdef float[:,::1] resampled_fib = np.zeros((n_pts,3), dtype=np.float32)
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

cdef void tot_lenght(float[:,::1] fib_in, float* length) nogil:
    cdef size_t i = 0

    length[0] = 0.0
    for i in xrange(1,fib_in.shape[0]):
        length[i] = length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 )


cdef float[:,::1] extract_ending_pts(float[:,::1] fib_in, float[:,::1] resampled_fib) :
    cdef int nb_pts_in = fib_in.shape[0]
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    resampled_fib[1][0] = fib_in[nb_pts_in-1][0]
    resampled_fib[1][1] = fib_in[nb_pts_in-1][1]
    resampled_fib[1][2] = fib_in[nb_pts_in-1][2]

    return resampled_fib

cdef float[:,::1] set_number_of_points(float[:,::1] fib_in, int nb_pts, float[:,::1] resampled_fib) nogil:
    cdef float[:] start = fib_in[0]
    cdef int nb_pts_in = fib_in.shape[0]
    cdef float[:] end = fib_in[nb_pts_in-1]
    cdef float* vers = <float*>malloc(3*sizeof(float))
    cdef float* lenghts = <float*>malloc(fib_in.shape[0]*sizeof(float))
    cdef size_t i = 0
    cdef size_t j = 0
    cdef float sum_step = 0
    tot_lenght(fib_in, lenghts)

    cdef float step_size = lenghts[nb_pts_in-1]/(nb_pts-1)
    cdef float sum_len = 0
    cdef float ratio = 0

    # for i in xrange(1, lenghts.shape[0]-1):
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    while sum_step < lenghts[nb_pts_in-1]:
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

    free(vers)
    free(lenghts)
    return resampled_fib


cdef (int, int) compute_dist(float[:,::1] fib_in, float[:,:,::1] target, float thr,
                            float d1_x, float d1_y, float d1_z, int num_c, int num_pt) nogil:
    """Compute the distance between a fiber and a set of centroids"""
    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef float maxdist_fib = 10000000000
    cdef int  i = 0
    cdef int  j = 0
    cdef int fib_idx = 0
    cdef int idx_ret = 0
    cdef int flipped_temp = 0
    cdef int flipped = 0

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
            maxdist_pt = maxdist_pt_d/num_pt
            flipped_temp = 0
        else:
            maxdist_pt = maxdist_pt_i/num_pt
            flipped_temp = 1
        
        if maxdist_pt < maxdist_fib:
            maxdist_fib = maxdist_pt
            flipped = flipped_temp
            idx_ret = i
    if maxdist_fib < thr:
        return (idx_ret, flipped)

    return (num_c, flipped)


cpdef cluster(filename_in: str, save_assignments: str, output_folder: str,
              threshold: float=10.0, n_pts: int=10, split: bool=False,
              force: bool=False, verbose: bool=False):
    """ Cluster streamlines in a tractogram based on average euclidean distance.
    TODO: DOCUMENTATION
    """
    ui.INFO(f"\n\nQB v2.0 clustering thr: {threshold}, pts: {n_pts}")
    # ui.set_verbose( 2 if verbose else 1 )
    ui.set_verbose( 2 )
    if not os.path.isfile(filename_in):
        ui.ERROR( f'File "{filename_in}" not found' )


    if np.isscalar( threshold ) :
        threshold = threshold
    
    cdef LazyTractogram TCK_in = LazyTractogram( filename_in, mode='r' )
    

    tractogram_gen = nib.streamlines.load(filename_in, lazy_load=True)
    cdef int n_streamlines = int( TCK_in.header['count'] )
    ui.INFO( f'  - {n_streamlines} streamlines found' )

    cdef int nb_pts = n_pts
    cdef float[:,::1] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:,::1] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
    cdef float [:,::1] s0 = np.empty( (1000, 3), dtype=np.float32 )
    cdef float [:,::1] s = np.empty( (1000, 3), dtype=np.float32 )
    TCK_in._read_streamline() 
    s0 = set_number_of_points(TCK_in.streamline[:TCK_in.n_pts], nb_pts, resampled_fib)
    # s0 = set_number_of_points(next(tractogram_gen.streamlines), nb_pts, resampled_fib)

    cdef float [:,::1] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,::1] streamline_in = np.zeros((nb_pts, 3), dtype=np.float32)
    cdef float[:,::1] streamline_in_stp = np.zeros((nb_pts, 3), dtype=np.float32)
    cdef int[:] c_w = np.ones(n_streamlines, dtype=np.int32)
    cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_stream_in = np.zeros(3, dtype=np.float32)
    cdef float [:] new_p_centr = np.zeros(3, dtype=np.float32)
    cdef size_t  i = 0
    cdef size_t  s_i = 0
    cdef size_t  p = 0
    cdef size_t  n_i = 0
    cdef float thr = threshold
    cdef int t = 0
    cdef int new_c = 1
    cdef int flipped = 0
    cdef int weight_centr = 0

    cdef float d1_x, d1_y, d1_z
    cdef float dt, dm1_d, dm1_i, dm2, dm3



    set_centroids[0] = s0
    cdef int [:] clust_idx = np.zeros(n_streamlines, dtype=np.int32)
    t1 = time.time()
    # with gil:
    if True:
        for i in xrange(n_streamlines):
        # for i, s in enumerate(tractogram_gen.streamlines):
        # for i in xrange(100):
            TCK_in._read_streamline()
            for s_i in xrange(TCK_in.n_pts):
                s[s_i] = TCK_in.streamline[s_i]
            # s = TCK_in.streamline[:TCK_in.n_pts]
            # with gil: print(TCK_in.n_pts)
            # print(f"i:{i}, # clusters:{new_c}", end="\r")
            # streamline_in[:] = extract_ending_pts(s, resampled_fib)
            streamline_in[:] = set_number_of_points(s[:TCK_in.n_pts], nb_pts, resampled_fib)

            t, flipped = compute_dist(streamline_in, set_centroids[:new_c], thr, d1_x, d1_y, d1_z, new_c, nb_pts)

            clust_idx[i]= t
            # with gil: print(f"cluster: {t}")
            weight_centr = c_w[t]
            if t < new_c:
                if flipped:
                    for p in xrange(nb_pts):
                        pt_centr = set_centroids[t][p]
                        pt_stream_in = streamline_in[nb_pts-p-1]
                        new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                        new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                        new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                        new_centroid[p] = new_p_centr
                else:
                    for p in xrange(nb_pts):
                        pt_centr = set_centroids[t][p]
                        pt_stream_in = streamline_in[p]
                        new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                        new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                        new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                        new_centroid[p] = new_p_centr
                c_w[t] += 1

            else:
                for n_i in xrange(nb_pts):
                    new_centroid[n_i] = streamline_in[n_i]
                new_c += 1

            set_centroids[t] = new_centroid

    # print(f"time required: {np.round((time.time()-t1)/60, 3)} minutes")
    # print(f"total_number of streamlines: {len(clust_idx)}")
    # print(f"number of clusters {len(np.unique(clust_idx))}")
    if TCK_in is not None:
        TCK_in.close()
    return clust_idx