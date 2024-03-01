#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

"""Functions to perform clustering of tractograms"""

import os, sys
import numpy as np
cimport numpy as np
from dicelib.lazytractogram cimport LazyTractogram
from dicelib.connectivity import assign
from dicelib.tractogram import split as split_bundles
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import time
from concurrent.futures import ThreadPoolExecutor as tdp
import concurrent.futures as cf
from dicelib import ui
import shutil



cdef void tot_lenght(float[:,::1] fib_in, float* length) noexcept nogil:
    cdef size_t i = 0

    length[0] = 0.0
    for i in xrange(1,fib_in.shape[0]):
        length[i] = <float>(length[i-1]+ sqrt( (fib_in[i][0]-fib_in[i-1][0])**2 + (fib_in[i][1]-fib_in[i-1][1])**2 + (fib_in[i][2]-fib_in[i-1][2])**2 ))


cdef float[:,::1] extract_ending_pts(float[:,::1] fib_in, float[:,::1] resampled_fib) :
    cdef int nb_pts_in = fib_in.shape[0]
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    resampled_fib[1][0] = fib_in[nb_pts_in-1][0]
    resampled_fib[1][1] = fib_in[nb_pts_in-1][1]
    resampled_fib[1][2] = fib_in[nb_pts_in-1][2]

    return resampled_fib


cdef void set_number_of_points(float[:,::1] fib_in, int nb_pts, float[:,::1] resampled_fib, float *vers, float *lengths) noexcept nogil:
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

    # free(vers)
    # free(lengths)


cdef (int, int) compute_dist_mean(float[:,::1] fib_in, float[:,:,::1] target, float thr,
                            float d1_x, float d1_y, float d1_z, int num_c, int num_pt) noexcept nogil:
    """Compute the distance between a fiber and a set of centroids"""
    cdef float meandist_pt   = 0
    cdef float meandist_pt_d = 0
    cdef float meandist_pt_i = 0
    cdef float meandist_fib = 10000000000
    cdef int  i = 0
    cdef int  j = 0
    cdef int idx_ret = 0
    cdef int flipped_temp = 0
    cdef int flipped = 0

    for i in xrange(num_c):
        meandist_pt_d = 0
        meandist_pt_i = 0

        for j in xrange(num_pt):

            d1_x = (target[i][j][0] - fib_in[j][0])**2
            d1_y = (target[i][j][1] - fib_in[j][1])**2
            d1_z = (target[i][j][2] - fib_in[j][2])**2

            meandist_pt_d += sqrt(d1_x + d1_y + d1_z)


            d1_x = (target[i][j][0] - fib_in[num_pt-j-1][0])**2
            d1_y = (target[i][j][1] - fib_in[num_pt-j-1][1])**2
            d1_z = (target[i][j][2] - fib_in[num_pt-j-1][2])**2
            
            meandist_pt_i += sqrt(d1_x + d1_y + d1_z)
        if meandist_pt_d < meandist_pt_i:
            meandist_pt = meandist_pt_d/num_pt
            flipped_temp = 0
        else:
            meandist_pt = meandist_pt_i/num_pt
            flipped_temp = 1
        
        if meandist_pt < meandist_fib:
            meandist_fib = meandist_pt
            flipped = flipped_temp
            idx_ret = i
    if meandist_fib < thr:
        return (idx_ret, flipped)

    return (num_c, flipped)


cdef (int, int) compute_dist_max(float[:,::1] fib_in, float[:,:,::1] target, float thr,
                            float d1_x, float d1_y, float d1_z, int num_c, int num_pt) noexcept nogil:
    """Compute the distance between a fiber and a set of centroids"""
    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef float maxdist_fib  = 10000000000
    cdef int  i = 0
    cdef int  j = 0
    cdef int idx_ret = 0
    cdef int flipped_temp = 0
    cdef int flipped = 0

    for i in xrange(num_c):
        maxdist_pt_d = 0
        maxdist_pt_i = 0
        maxdist_pt = 0

        for j in xrange(num_pt):

            d1_x = (target[i][j][0] - fib_in[j][0])**2
            d1_y = (target[i][j][1] - fib_in[j][1])**2
            d1_z = (target[i][j][2] - fib_in[j][2])**2

            maxdist_pt_d = sqrt(d1_x + d1_y + d1_z)


            d1_x = (target[i][j][0] - fib_in[num_pt-j-1][0])**2
            d1_y = (target[i][j][1] - fib_in[num_pt-j-1][1])**2
            d1_z = (target[i][j][2] - fib_in[num_pt-j-1][2])**2
            
            maxdist_pt_i = sqrt(d1_x + d1_y + d1_z)

            if maxdist_pt_d < maxdist_pt_i and maxdist_pt_d > maxdist_pt:
                maxdist_pt = maxdist_pt_d
                flipped_temp = 0
            elif maxdist_pt_d > maxdist_pt_i and maxdist_pt_i > maxdist_pt:
                maxdist_pt = maxdist_pt_i
                flipped_temp = 1
        
        if maxdist_pt < maxdist_fib:
            maxdist_fib = maxdist_pt
            flipped = flipped_temp
            idx_ret = i
    if maxdist_fib < thr:
        return (idx_ret, flipped)

    return (num_c, flipped)


cpdef float [:] compute_dist_centroid(float[:,:,::1] centroids, int [:] clust_idx, str path_resampled, int num_pt):
    """Compute the distance between the streamlines and the centroid of the cluster to which they belong
        centroids      = array with the final centroids
        clust_idx      = array containing for each streamline the idx of the cluster to which it belongs
        path_resampled = path of the input streamlines after resampling
        num_pt         = number of points
    """
    cdef float dist_d = 0
    cdef float dist_f = 0
    cdef float d_x = 0
    cdef float d_y = 0
    cdef float d_z = 0
    cdef size_t  i = 0
    cdef size_t  j = 0

    cdef LazyTractogram TCK_res = LazyTractogram( path_resampled, mode='r' )
    cdef int num_str = int( TCK_res.header['count'] )
    cdef float [:] distances = np.zeros(num_str, dtype=np.float32) # array containing for each streamline the distance from the centroid (output)

    for i in xrange(num_str):
        TCK_res.read_streamline()
        dist_d = 0
        dist_f = 0

        for j in xrange(num_pt):
            # direct
            d_x = (centroids[clust_idx[i]][j][0] - TCK_res.streamline[j][0])**2
            d_y = (centroids[clust_idx[i]][j][1] - TCK_res.streamline[j][1])**2
            d_z = (centroids[clust_idx[i]][j][2] - TCK_res.streamline[j][2])**2
            dist_d += sqrt(d_x + d_y + d_z)

            # flipped
            d_x = (centroids[clust_idx[i]][j][0] - TCK_res.streamline[num_pt-j-1][0])**2
            d_y = (centroids[clust_idx[i]][j][1] - TCK_res.streamline[num_pt-j-1][1])**2
            d_z = (centroids[clust_idx[i]][j][2] - TCK_res.streamline[num_pt-j-1][2])**2
            dist_f += sqrt(d_x + d_y + d_z)

        if dist_d < dist_f:
            distances[i] = dist_d/num_pt
        else:
            distances[i] = dist_f/num_pt

    return distances


cpdef cluster(filename_in: str, metric: str="mean", threshold: float=10.0, n_pts: int=10,
              verbose: int=2):
    """ Cluster streamlines in a tractogram based on average euclidean distance.

    Parameters
    ----------
    filename_in : str
        Path to the input tractogram file.
    threshold : float, optional
        Threshold for the clustering.
    n_pts : int, optional
        Number of points to resample the streamlines to.
    verbose : bool, optional
        Whether to print out additional information during the clustering.
    """

    if not os.path.isfile(filename_in):
        ui.ERROR( f'File "{filename_in}" not found' )


    if np.isscalar( threshold ) :
        threshold = threshold
    
    cdef LazyTractogram TCK_in = LazyTractogram( filename_in, mode='r', max_points=1000 )
    ui.set_verbose( verbose )

    # tractogram_gen = nib.streamlines.load(filename_in, lazy_load=True)
    cdef int n_streamlines = int( TCK_in.header['count'] )
    if n_streamlines == 0: return
    ui.INFO( f'  - {n_streamlines} streamlines found' )

    cdef int nb_pts = n_pts
    cdef bool metric_mean = metric == 'mean'
    cdef float[:,::1] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:,::1] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
    cdef float[:,::1] s0 = np.empty( (n_pts, 3), dtype=np.float32 )
    cdef float* vers = <float*>malloc(3*sizeof(float))
    cdef float* lengths = <float*>malloc(1000*sizeof(float))
    TCK_in._read_streamline() 
    cdef size_t pp = 0

    if TCK_in.n_pts == nb_pts: # no need to resample
        for pp in xrange(nb_pts): # copy streamline
            s0[pp][0] = TCK_in.streamline[pp][0]
            s0[pp][1] = TCK_in.streamline[pp][1]
            s0[pp][2] = TCK_in.streamline[pp][2]
    else:
        set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], nb_pts, s0, vers, lengths)

    cdef float[:,::1] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,::1] streamline_in = np.zeros((nb_pts,3), dtype=np.float32)
    cdef int[:] c_w = np.ones(n_streamlines, dtype=np.int32)
    cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_stream_in = np.zeros(3, dtype=np.float32)
    cdef float[:] new_p_centr = np.zeros(3, dtype=np.float32)
    cdef size_t  i = 0
    cdef size_t  p = 0
    cdef size_t  n_i = 0
    cdef float thr = threshold
    cdef int t = 0
    cdef int new_c = 1
    cdef int flipped = 0
    cdef int weight_centr = 0
    cdef float d1_x = 0
    cdef float d1_y = 0
    cdef float d1_z= 0


    set_centroids[0] = s0
    cdef int [:] clust_idx = np.zeros(n_streamlines, dtype=np.int32)
    t1 = time.time()
 
    with ui.ProgressBar(total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
        for i in xrange(1, n_streamlines, 1):
            TCK_in._read_streamline()
            if TCK_in.n_pts == nb_pts: # no need to resample
                for pp in xrange(nb_pts): # copy streamline
                    streamline_in[pp][0] = TCK_in.streamline[pp][0]
                    streamline_in[pp][1] = TCK_in.streamline[pp][1]
                    streamline_in[pp][2] = TCK_in.streamline[pp][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], nb_pts, streamline_in[:] , vers, lengths)

            if metric_mean:
                t, flipped = compute_dist_mean(streamline_in, set_centroids[:new_c], thr, d1_x, d1_y, d1_z, new_c, nb_pts)
            else:
                t, flipped = compute_dist_max(streamline_in, set_centroids[:new_c], thr, d1_x, d1_y, d1_z, new_c, nb_pts)

            clust_idx[i]= t
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
            pbar.update()
    
    if TCK_in is not None:
        TCK_in.close()
    return clust_idx, set_centroids[:new_c]


cpdef closest_streamline(file_name_in: str, float[:,:,::1] target, int [:] clust_idx, int num_pt, int num_c, int [:] centr_len, verbose: int=2):
    """
    Compute the distance between a fiber and a set of centroids
    
    Parameters
    ----------
    file_name_in : str
        Path to the input tractogram file.
    target : float[:,:,::1]
        Centroids to compare the streamlines to.
    clust_idx : int[:]
        Cluster assignments for each streamline.
    num_pt : int
        Number of points to resample the streamlines to.
    num_c : int
        Number of centroids.
    centr_len : int[:]
        Length of each centroid.
    """

    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef size_t  i_f = 0
    cdef int  j = 0
    # cdef int  c_i = 0
    cdef float d1_x = 0
    cdef float d1_y = 0
    cdef float d1_z= 0
    cdef float d2_x = 0
    cdef float d2_y = 0
    cdef float d2_z= 0
    cdef float [:] fib_centr_dist = np.repeat(1000, num_c).astype(np.float32)
    cdef float[:,::1] fib_in = np.zeros((num_pt,3), dtype=np.float32)
    cdef float[:,::1] resampled_fib = np.zeros((num_pt,3), dtype=np.float32)
    cdef float [:,:,::1] centroids = np.zeros((num_c, 3000,3), dtype=np.float32)
    cdef LazyTractogram TCK_in = LazyTractogram( file_name_in, mode='r' )
    cdef int n_streamlines = int( TCK_in.header['count'] )
    cdef float* vers = <float*>malloc(3*sizeof(float))
    cdef float* lengths = <float*>malloc(1000*sizeof(float))
    cdef size_t p = 0

    
    with ui.ProgressBar(total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
        for i_f in xrange(n_streamlines):
            TCK_in._read_streamline()
            c_i = clust_idx[i_f]
            if TCK_in.n_pts == num_pt: # no need to resample
                for p in xrange(num_pt): # copy streamline
                    fib_in[p][0] = TCK_in.streamline[p][0]
                    fib_in[p][1] = TCK_in.streamline[p][1]
                    fib_in[p][2] = TCK_in.streamline[p][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], num_pt, fib_in[:] , vers, lengths)
            maxdist_pt_d = 0
            maxdist_pt_i = 0

            for j in xrange(num_pt):

                d1_x = (fib_in[j][0] - target[c_i][j][0])**2
                d1_y = (fib_in[j][1] - target[c_i][j][1])**2
                d1_z = (fib_in[j][2] - target[c_i][j][2])**2

                maxdist_pt_d += sqrt(d1_x + d1_y + d1_z)

                d2_x = (fib_in[j][0] - target[c_i][num_pt-j-1][0])**2
                d2_y = (fib_in[j][1] - target[c_i][num_pt-j-1][1])**2
                d2_z = (fib_in[j][2] - target[c_i][num_pt-j-1][2])**2
                
                maxdist_pt_i += sqrt(d2_x + d2_y + d2_z)
            if maxdist_pt_d < maxdist_pt_i:
                maxdist_pt = maxdist_pt_d/num_pt
            else:
                maxdist_pt = maxdist_pt_i/num_pt
            
            if maxdist_pt < fib_centr_dist[c_i]:
                fib_centr_dist[c_i] = maxdist_pt
                centroids[c_i, :TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts].copy()
                centr_len[c_i] = TCK_in.n_pts
            pbar.update()

    if TCK_in is not None:
        TCK_in.close()

    return centroids


cpdef cluster_chunk(filenames: list[str], threshold: float=10.0, n_pts: int=10, metric: str="mean"):
    """ Cluster streamlines in a tractogram based on average euclidean distance.

    Parameters
    ----------
    filenames : list[str]
        List of paths to the input tractogram files.
    threshold : float, optional
        Threshold for the clustering.
    n_pts : int, optional
        Number of points to resample the streamlines to.

    """

    cdef float[:,:,:,::1] set_centroids = np.zeros((len(filenames), 100000, n_pts, 3), dtype=np.float32)
    cdef LazyTractogram TCK_in
    cdef int [:] n_streamlines = np.zeros(len(filenames), dtype=np.int32)
    cdef int [:] header_params = np.zeros(len(filenames), dtype=np.intc)
    cdef float[:,:,::1] resampled_fib = np.zeros((1,n_pts,3), dtype=np.float32)
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t pp = 0

    idx_cl = np.zeros((len(filenames), 100000), dtype=np.intc)
    cdef int[:,::1] idx_closest = idx_cl
    cdef float* vers = <float*>malloc(3*sizeof(float))
    cdef float* lengths = <float*>malloc(1000*sizeof(float))

    for i, filename in enumerate(filenames):
        TCK_in = LazyTractogram( filename, mode='r', max_points=1000 )
        idx = np.load(f'{filename[:len(filename)-4]}.npy').astype(np.intc)
        idx_cl[i, :idx.shape[0]] = idx
        n_streamlines[i] = int(TCK_in.header['count'])
        header_params[i] = int(TCK_in.header['file'][2:])
        TCK_in._read_streamline()
        if TCK_in.n_pts == n_pts: # no need to resample
            for pp in xrange(n_pts): # copy streamline
                set_centroids[i, 0, pp, 0] = TCK_in.streamline[pp][0]
                set_centroids[i, 0, pp, 1] = TCK_in.streamline[pp][1]
                set_centroids[i, 0, pp, 2] = TCK_in.streamline[pp][2]
        else:
            set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], n_pts, set_centroids[i, 0], vers, lengths)
        TCK_in.close()


    in_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines)), 1000, 3), dtype=np.float32)
    
    cdef float[:,:,:,::1] in_streamlines_view = in_streamlines
    cdef int [:,::1] len_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef float[:,:,:,::1] resampled_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines)), n_pts, 3), dtype=np.float32)

    for i, filename in enumerate(filenames):
        TCK_in = LazyTractogram( filename, mode='r', max_points=1000 )
        for st in range(n_streamlines[i]):
            TCK_in._read_streamline()
            in_streamlines[i][st][:TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts]
            len_streamlines[i][st] = TCK_in.n_pts
            if TCK_in.n_pts == n_pts: # no need to resample
                for pp in xrange(n_pts): # copy streamline
                    resampled_streamlines[i, st, pp, 0] = TCK_in.streamline[pp][0]
                    resampled_streamlines[i, st, pp, 1] = TCK_in.streamline[pp][1]
                    resampled_streamlines[i, st, pp, 2] = TCK_in.streamline[pp][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], n_pts, resampled_streamlines[i, st], vers, lengths)
        TCK_in.close()
    free(vers)
    free(lengths)
    
    cdef int nb_pts = n_pts
    idx_cl_return = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.intc)
    cdef int[:,::1] idx_closest_return = idx_cl_return
    cdef float [:,::1] new_centroid = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:] fib_centr_dist = np.zeros((len(filenames), int(np.max(n_streamlines)))).astype(np.float32)
    fib_centr_dist[:] = 1000
    clst_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines)), 1000, 3), dtype=np.float32)
    cdef float[:,:,:,::1] clst_streamlines_view = clst_streamlines
    cdef float[:,::1] streamline_in = np.zeros((nb_pts, 3), dtype=np.float32)
    cdef int[:,::1] c_w = np.ones((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef float[:] pt_centr = np.zeros(3, dtype=np.float32)
    cdef float[:] pt_stream_in = np.zeros(3, dtype=np.float32)
    cdef float [:] new_p_centr = np.zeros(3, dtype=np.float32)
    centr_len = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef int [:,:] centr_len_view = centr_len
    cdef size_t  p = 0
    cdef size_t  n_i = 0
    cdef float thr = threshold
    cdef int t = 0
    cdef int c_i = 0
    new_c = np.ones(len(filenames), dtype=np.int32)
    cdef int [:] new_c_view = new_c
    cdef int flipped = 0
    cdef int weight_centr = 0
    cdef float d1_x = 0
    cdef float d1_y = 0
    cdef float d1_z = 0
    cdef int [:,:] clust_idx = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef bool metric_mean = metric == 'mean'
    
    with nogil:
        for i in range(in_streamlines_view.shape[0]):
            for j in range(1, n_streamlines[i], 1):
                if metric_mean:
                    t, flipped = compute_dist_mean(resampled_streamlines[i, j], set_centroids[i,:new_c_view[i]], thr, d1_x, d1_y, d1_z, new_c_view[i], nb_pts)
                else:
                    t, flipped = compute_dist_max(resampled_streamlines[i, j], set_centroids[i,:new_c_view[i]], thr, d1_x, d1_y, d1_z, new_c_view[i], nb_pts)

                clust_idx[i,j]= t
                weight_centr = c_w[i,t]
                if t < new_c_view[i]:
                    if flipped:
                        for p in xrange(nb_pts):
                            pt_centr = set_centroids[i,t,p]
                            pt_stream_in = resampled_streamlines[i, j][nb_pts-p-1]
                            new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                            new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                            new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                            new_centroid[p] = new_p_centr
                    else:
                        for p in xrange(nb_pts):
                            pt_centr = set_centroids[i,t,p]
                            pt_stream_in = resampled_streamlines[i, j][p]
                            new_p_centr[0] = (weight_centr * pt_centr[0] + pt_stream_in[0])/(weight_centr+1)
                            new_p_centr[1] = (weight_centr * pt_centr[1] + pt_stream_in[1])/(weight_centr+1)
                            new_p_centr[2] = (weight_centr * pt_centr[2] + pt_stream_in[2])/(weight_centr+1)
                            new_centroid[p] = new_p_centr
                    c_w[i,t] += 1

                else:
                    for n_i in xrange(nb_pts):
                        new_centroid[n_i] = resampled_streamlines[i, j][n_i]
                    new_c_view[i] += 1
                set_centroids[i,t] = new_centroid

        for i in range(in_streamlines_view.shape[0]):
            for j in range(n_streamlines[i]):
                c_i = clust_idx[i,j]
                closest_streamline_s( in_streamlines_view[i,j,:len_streamlines[i][j]], len_streamlines[i][j], c_i,
                                     set_centroids[i, c_i], resampled_streamlines[i, j], nb_pts, centr_len_view[i],
                                     fib_centr_dist[i], clst_streamlines_view[i], idx_closest[i], idx_closest_return[i], j)


    return clst_streamlines, centr_len, new_c, idx_cl_return
    


cdef void closest_streamline_s( float[:,::1] streamline_in, int n_pts, int c_i, float[:,::1] target, float[:,::1] fib_in,
                                int nb_pts, int [:] centr_len, float[:] fib_centr_dist, float[:,:,::1] closest_streamlines,
                                int[:] idx_closest, int[:] idx_closest_return, int jj) noexcept nogil:
    cdef float maxdist_pt   = 0
    cdef float maxdist_pt_d = 0
    cdef float maxdist_pt_i = 0
    cdef float d1_x = 0
    cdef float d1_y = 0
    cdef float d1_z= 0
    cdef float d2_x = 0
    cdef float d2_y = 0
    cdef float d2_z= 0
    cdef int  j = 0

    maxdist_pt_d = 0
    maxdist_pt_i = 0

    for j in range(nb_pts):

        d1_x = (fib_in[j][0] - target[j][0])**2
        d1_y = (fib_in[j][1] - target[j][1])**2
        d1_z = (fib_in[j][2] - target[j][2])**2

        maxdist_pt_d += sqrt(d1_x + d1_y + d1_z)

        d2_x = (fib_in[j][0] - target[nb_pts-j-1][0])**2
        d2_y = (fib_in[j][1] - target[nb_pts-j-1][1])**2
        d2_z = (fib_in[j][2] - target[nb_pts-j-1][2])**2

        maxdist_pt_i += sqrt(d2_x + d2_y + d2_z)

    if maxdist_pt_d < maxdist_pt_i:
        maxdist_pt = maxdist_pt_d/nb_pts
    else:
        maxdist_pt = maxdist_pt_i/nb_pts
    
    if maxdist_pt < fib_centr_dist[c_i]: 
        fib_centr_dist[c_i] = maxdist_pt
        copy_s(streamline_in, closest_streamlines[c_i], n_pts)
        centr_len[c_i] = n_pts
        idx_closest_return[c_i] = idx_closest[jj]


cdef void copy_s(float[:,::1] fib_in, float[:,::1] fib_out, int n_pts) noexcept nogil:
    cdef size_t i = 0
    for i in range(n_pts):
        fib_out[i][0] = fib_in[i][0]
        fib_out[i][1] = fib_in[i][1]
        fib_out[i][2] = fib_in[i][2]



def run_clustering(file_name_in: str, file_name_out: str, output_folder: str=None, atlas: str=None, conn_thr: float=2.0,
                    clust_thr: float=2.0, metric: str="mean", n_pts: int=10,
                    n_threads: int=None, force: bool=False, verbose: int=2, keep_temp_files: bool=False):
    """ Cluster streamlines in a tractogram based on average euclidean distance.

    Parameters
    ----------
    file_name_in : str
        Path to the input tractogram file.
    output_folder : str
        Path to the output folder. If None, files are saved in the input tractogram folder.
    atlas : str, optional
        Path to the atlas file.
    conn_thr : float, optional
        Threshold for the connectivity assignment (default: 2.0).
    clust_thr : float, optional
        Threshold for the clustering (default: 2.0).
    metric : str, optional
        Metric to use for the clustering. Either "mean" or "max" (default: "mean").
    n_pts : int, optional
        Number of points to resample the streamlines to (default: 10).
    n_threads : int, optional
        Number of threads to use for the clustering.
    force : bool, optional
        Whether to overwrite existing files.
    verbose : bool, optional
        Whether to print out additional information during the clustering.
    keep_temp_files : bool, optional
        Whether to keep temporary files.
    """

    ui.set_verbose(verbose)

    ui.INFO(f"  - Clustering with threshold: {clust_thr}, using {n_pts} points")

    def compute_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


    MAX_THREAD = 3

    TCK_in = LazyTractogram( file_name_in, mode='r' )

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if not os.path.isabs(file_name_out) and output_folder is None:
        output_folder = os.getcwd()
        file_name_out = os.path.join(output_folder, file_name_out)
    elif os.path.isabs(file_name_out) and output_folder is None:
        output_folder = os.path.dirname(file_name_out)
    elif not os.path.isabs(file_name_out) and output_folder is not None:
        file_name_out = os.path.join(os.getcwd(), file_name_out)
    
        

    # check if file exists
    if os.path.isfile(file_name_out) and not force:
        ui.ERROR( 'Output tractogram file already exists, use -f to overwrite' )
        return
    TCK_out = LazyTractogram(file_name_out, mode='w', header=TCK_in.header )
    num_streamlines = int(TCK_in.header["count"])

    if atlas:
        chunk_size = int(num_streamlines/MAX_THREAD)
        chunk_groups = [e for e in compute_chunks( np.arange(num_streamlines),chunk_size)]

        # check if save_assignments is None
        save_assignments = os.path.join(output_folder, f'{os.path.basename(file_name_in)[:len(file_name_in)-4]}_assignments.txt')
        temp_idx_arr = np.arange(num_streamlines)
        temp_idx = os.path.join(output_folder, 'streamline_idx.npy')
        np.save( temp_idx, temp_idx_arr )

        chunks_asgn = []
        t0 = time.time()

        pbar_array = np.zeros(MAX_THREAD, dtype=np.int32)

        with ui.ProgressBar( multithread_progress=pbar_array, total=num_streamlines, disable=(verbose in [0,1,3]), hide_on_exit=True) as pbar:
            with tdp(max_workers=MAX_THREAD) as executor:
                future = [executor.submit( assign, file_name_in, pbar_array, i, start_chunk=int(chunk_groups[i][0]),
                                            end_chunk=int(chunk_groups[i][len(chunk_groups[i])-1]+1),
                                            gm_map_file=atlas, threshold=conn_thr ) for i in range(len(chunk_groups))]
                chunks_asgn = [f.result() for f in future]
                chunks_asgn = [c for f in chunks_asgn for c in f]

        t1 = time.time()
        ui.INFO(f"  - Time taken for connectivity: {t1-t0}")
        out_assignment_ext = os.path.splitext(save_assignments)[1]

        if out_assignment_ext not in ['.txt', '.npy']:
            ui.ERROR(f"  - Invalid extension for the output scalar file" )
        if os.path.isfile(save_assignments) and not force:
            ui.ERROR(f"  - Output scalar file already exists, use -f to overwrite" )

        if out_assignment_ext=='.txt':
            with open(save_assignments, "w") as text_file:
                for reg in chunks_asgn:
                    print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
        else:
            np.save( save_assignments, chunks_asgn, allow_pickle=False )

        t0 = time.time()
        output_bundles_folder = os.path.join(output_folder, 'bundles')
        split_bundles(input_tractogram=file_name_in, input_assignments=save_assignments, output_folder=output_bundles_folder,
                      weights_in=temp_idx, force=force)
        t1 = time.time()
        ui.set_verbose(verbose)
        ui.INFO(f"  - Time bundles splitting: {t1-t0}")
        
        bundles = {}
        for  dirpath, _, filenames in os.walk(output_bundles_folder):
            for i, f in enumerate(filenames):
                if f.endswith('.tck') and not f.startswith('unassigned'):
                    filename = os.path.abspath(os.path.join(dirpath, f))
                    bundles[i] = (filename, os.path.getsize(filename))

        if n_threads:
            MAX_THREAD = n_threads
        else:
            MAX_THREAD = os.cpu_count()

        # NOTE: optimal
        MAX_THREAD = 6


        ref_indices = []
        TCK_out_size = 0

        MAX_BYTES = 2e8
        executor = tdp(max_workers=MAX_THREAD)
        t0 = time.time()
        chunk_list = []

        # NOTE: compute chunks
        while len(bundles.items()) > 0:
            to_delete = []
            new_chunk = []
            for k, bundle in bundles.items():
                new_chunk_size = [os.path.getsize(f) for f in new_chunk]
                new_chunk_size.append(bundle[1])
                if max(new_chunk_size)*len(new_chunk) < MAX_BYTES:
                    new_chunk.append(bundle[0])
                    to_delete.append(k)
            [bundles.pop(k) for k in to_delete]
            chunk_list.append(new_chunk)
                
        with ui.ProgressBar(total=len(chunk_list), disable=(verbose in [0,1,3]), hide_on_exit=True) as pbar:
            future = [executor.submit(cluster_chunk,
                                        chunk,
                                        clust_thr,
                                        n_pts=n_pts,
                                        metric=metric) for chunk in chunk_list]
            for i, f in enumerate(cf.as_completed(future)):
                bundle_new_c, bundle_centr_len, bundle_num_c, idx_clst = f.result()

                for i_b in range(len(bundle_num_c)):
                    ref_indices.extend(idx_clst[i_b][:bundle_num_c[i_b]].tolist())
                    new_centroids, new_centroids_len = bundle_new_c[i_b], bundle_centr_len[i_b]
                    for i_s in range(bundle_num_c[i_b]):
                        TCK_out.write_streamline(new_centroids[i_s, :new_centroids_len[i_s]], new_centroids_len[i_s] )
                        TCK_out_size += 1
                pbar.update()
            TCK_out.close( write_eof=True, count= TCK_out_size)

        t1 = time.time()
        ui.INFO(f"  - Time taken to cluster and find closest streamlines: {t1-t0}")
        ui.INFO(f"  - Number of computed centroids: {TCK_out_size}")

        if not keep_temp_files:
            shutil.rmtree(output_bundles_folder)
            os.remove(save_assignments)
        os.remove(temp_idx)

    else:
        t0 = time.time()

        hash_superset = np.empty( num_streamlines, dtype=int)

        for i in range(num_streamlines):
            TCK_in._read_streamline()
            hash_superset[i] = hash(np.array(TCK_in.streamline[:TCK_in.n_pts]).tobytes())
        TCK_in.close()


        clust_idx, set_centroids = cluster(file_name_in,
                                            metric=metric,
                                            threshold=clust_thr,
                                            n_pts=n_pts,
                                            verbose=verbose
                                            )
        centr_len = np.zeros(set_centroids.shape[0], dtype=np.intc)
        new_c = closest_streamline(file_name_in, set_centroids, clust_idx, n_pts, set_centroids.shape[0], centr_len)
        
        TCK_out_size = 0
        ref_indices = []
        for i, n_c in enumerate(new_c):
            hash_val = hash(np.array(n_c[:centr_len[i]]).tobytes())
            ref_indices.append( np.flatnonzero(hash_superset == hash_val)[0] )
            TCK_out.write_streamline(n_c[:centr_len[i]], centr_len[i] )
            TCK_out_size += 1
        TCK_out.close( write_eof=True, count= TCK_out_size)

        t1 = time.time()
        ui.INFO(f"  - Time taken to cluster and find closest streamlines: {t1-t0}" )
        ui.INFO(f"  - Number of computed centroids: {TCK_out_size}" )

    if TCK_in is not None:
        TCK_in.close()

    return ref_indices

