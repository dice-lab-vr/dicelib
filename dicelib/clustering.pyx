# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

"""Functions to perform clustering of tractograms"""

from dicelib.connectivity import _assign as assign
from dicelib.tractogram import info, split as split_bundles
from dicelib.streamline import length as streamline_length
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, Dir, File, Num, format_time

from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import shutil
from sys import getsizeof
import time

import nibabel as nib
import numpy as np
import psutil

from dicelib.tractogram cimport LazyTractogram

from libc.math cimport sqrt
from libc.stdlib cimport free, malloc
from libcpp cimport bool

logger = setup_logger('clustering')

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


cpdef cluster(filename_in: str, metric: str="mean", threshold: float=4.0, n_pts: int=12,
              verbose: int=3):
    """ Cluster streamlines in a tractogram based on a given metric (mean or max distance to the centroids)

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
        logger.error(f'File \'{filename_in}\' not found')


    if np.isscalar( threshold ) :
        threshold = threshold
    
    cdef LazyTractogram TCK_in = LazyTractogram( filename_in, mode='r', max_points=1000 )
    set_verbose('clustering', verbose)

    # tractogram_gen = nib.streamlines.load(filename_in, lazy_load=True)
    cdef int n_streamlines = int( TCK_in.header['count'] )
    if n_streamlines == 0: return

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
 
    with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=True) as pbar:
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


cpdef closest_streamline(tractogram_in: str, float[:,:,::1] target, int [:] clust_idx, int num_pt, int num_c, int [:] centr_len, verbose: int=3):
    """
    Compute the distance between a fiber and a set of centroids
    
    Parameters
    ----------
    tractogram_in : str
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
    cdef LazyTractogram TCK_in = LazyTractogram( tractogram_in, mode='r' )
    cdef int n_streamlines = int( TCK_in.header['count'] )
    cdef float* vers = <float*>malloc(3*sizeof(float))
    cdef float* lengths = <float*>malloc(1000*sizeof(float))
    cdef size_t p = 0

    
    with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=True) as pbar:
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


cpdef cluster_chunk(filenames: list[str], num_fibs: int, threshold: float=10.0, n_pts: int=10, metric: str="mean"):
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

    cdef float[:,:,:,::1] set_centroids = np.zeros((len(filenames), num_fibs, n_pts, 3), dtype=np.float32)
    cdef LazyTractogram TCK_in
    cdef int [:] n_streamlines = np.zeros(len(filenames), dtype=np.int32)
    cdef int [:] header_params = np.zeros(len(filenames), dtype=np.intc)
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t pp = 0

    idx_cl = np.zeros((len(filenames), num_fibs), dtype=np.intc)
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
    cdef int [:,::1] clust_idx = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef int [:] bundle_n_streamlines = np.zeros(len(filenames), dtype=np.int32)
    cdef bool metric_mean = metric == 'mean'
    
    with nogil:
        for i in range(in_streamlines_view.shape[0]):
            bundle_n_streamlines[i] = n_streamlines[i]
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


    return clst_streamlines, centr_len, new_c, idx_cl_return, clust_idx, bundle_n_streamlines
    


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



def run_clustering(tractogram_in: str, tractogram_out: str, temp_folder: str=None, atlas: str=None, conn_thr: float=2.0,
                    clust_thr: float=2.0, metric: str="mean", n_pts: int=12, weights_in: str=None, weights_metric: str="sum",
                    weights_out: str=None, n_threads: int=None, force: bool=False, max_open: int=None, verbose: int=3,
                    keep_temp_files: bool=False, max_bytes: int=0, log_list=None):
    """Cluster streamlines in a tractogram based on a given metric. Possible metrics are "mean" and "max" (default: "mean").

    Parameters
    ----------
    tractogram_in : str
        Path to the input tractogram file.
    temp_folder : str
        Path to the temporary folder used to store the intermediate files.
    atlas : str, optional
        Path to the atlas file used to split the streamlines into bundles for parallel clustering.
    conn_thr : float, optional
        Distance threshold used for hierarchical clustering (default: 2.0).
    clust_thr : float, optional
        Distance threshold used for the final clustering (default: 2.0).
    metric : str, optional
        Metric to use for the clustering. Either "mean" or "max" (default: "mean").
    n_pts : int, optional
        Number of points to resample the streamlines to (default: 10).
    weights_in : str, optional
        Path to the weights containing a scalar value for each streamline.
    weights_metric : str, optional
        Metric used to assign a weight to each resulting centroid. Either "min", "max", "median", "mean" or "sum" (default: "sum").
    weights_out : str, optional
        Path to the output weights file.
    n_threads : int, optional
        Number of threads to use for the clustering.
    force : bool, optional
        Whether to overwrite existing files.
    verbose : bool, optional
        Whether to print out additional information during the clustering.
    keep_temp_files : bool, optional
        Whether to keep temporary files.
    """

    set_verbose('clustering', verbose)

    files = [
        File(name='tractogram_in', type_='input', path=tractogram_in, ext=['.tck']),
        File(name='tractogram_out', type_='output', path=tractogram_out, ext=['.tck'])
    ]
    if atlas is not None:
        files.append(File(name='atlas', type_='input', path=atlas, ext=['.nii', '.nii.gz']))
    temp_folder = temp_folder if temp_folder is not None else os.path.join(os.getcwd(), 'tmp')
    dirs = [
        Dir(name='tmp_folder', path=temp_folder)
    ]
    if weights_in is not None:
        files.append(File(name='weights_in', type_='input', path=weights_in, ext=['.txt', '.npy']))
    if weights_out is not None:
        files.append(File(name='weights_out', type_='output', path=weights_out, ext=['.txt', '.npy']))
    nums = [
        Num(name='clust_thr', value=clust_thr, min_=0.0, include_min=False),
        Num(name='atlas_dist', value=conn_thr, min_=0.0, include_min=True),
        Num(name='n_pts', value=n_pts, min_=2)
    ]
    if n_threads is not None:
        nums.append(Num(name='n_threads', value=n_threads, min_=1))

    if weights_metric not in ['min', 'max', 'median', 'sum', 'mean']:
        logger.error(f'Option {weights_metric} not valid, please choose between min, max, median, mean or sum')
    check_params(files=files, dirs=dirs, nums=nums, force=force)
    
    tmp_dir_is_created = False
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
        tmp_dir_is_created = True
    
    # other checks
    if metric not in ['mean', 'max']:
        logger.error(f'Invalid metric, must be \'mean\' or \'max\'')

    def compute_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if n_threads:
        MAX_THREAD = n_threads
    else:
        MAX_THREAD = os.cpu_count()

    TCK_in = LazyTractogram(tractogram_in, mode='r')
    num_streamlines = int(TCK_in.header["count"])

    if atlas:
        chunk_size = int(num_streamlines/MAX_THREAD)
        chunk_groups = [e for e in compute_chunks( np.arange(num_streamlines),chunk_size)]

        # check if save_assignments is None
        save_assignments = os.path.join(temp_folder, f'{os.path.basename(tractogram_in)[:-4]}_assignments.txt')
        temp_idx_arr = np.arange(num_streamlines)
        temp_idx = os.path.join(temp_folder, 'streamline_idx.npy')
        np.save( temp_idx, temp_idx_arr )

        chunks_asgn = []
        t0 = time.time()

        pbar_array = np.zeros(MAX_THREAD, dtype=np.int32)

        logger.info('Dividing the streamlines into anatomical bundles')
        atlas_img = nib.load(atlas)
        atlas_data = atlas_img.get_fdata()
        atlas_dtype = atlas_img.header.get_data_dtype()
        if atlas_dtype.char not in ['b',' h', 'i', 'l', 'B', 'H', 'I', 'L']:
            warning_msg = f'Atlas data type is \'{atlas_dtype}\'. It is recommended to use an integer data type.'
            logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)
        logger.subinfo('Computing assignments', indent_lvl=1, indent_char='*', with_progress=verbose>2)
        with ProgressBar(multithread_progress=pbar_array, total=num_streamlines, disable=verbose < 3, hide_on_exit=True, subinfo=True) as pbar:
            with ThreadPoolExecutor(max_workers=MAX_THREAD) as executor:
                future = [
                    executor.submit(
                        assign,
                        tractogram_in,
                        pbar_array,
                        i,
                        start_chunk=int(chunk_groups[i][0]),
                        end_chunk=int(chunk_groups[i][len(chunk_groups[i])-1]+1),
                        gm_map_data=atlas_data,
                        gm_map_img=atlas_img,
                        threshold=conn_thr ) for i in range(len(chunk_groups))]
                chunks_asgn = [f.result() for f in future]
                chunks_asgn = [c for f in chunks_asgn for c in f]

        t1 = time.time()
        logger.subinfo(f'Number of regions: {np.max(np.array(chunks_asgn))}', indent_lvl=1, indent_char='*')
        logger.info( f'[ {format_time(t1 - t0)} ]' )

        out_assignment_ext = os.path.splitext(save_assignments)[1]
        if out_assignment_ext not in ['.txt', '.npy']:
            logger.error(f'Invalid extension for the output scalar file')
        if os.path.isfile(save_assignments) and not force:
            logger.error(f'Output scalar file already exists, use -f to overwrite')

        if out_assignment_ext=='.txt':
            with open(save_assignments, "w") as text_file:
                for reg in chunks_asgn:
                    print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
        else:
            np.save( save_assignments, chunks_asgn, allow_pickle=False )

        t0 = time.time()
        output_bundles_folder = os.path.join(temp_folder, 'bundles')
        logger.info('Splitting the bundles into separate files')
        with ProgressBar(disable=verbose<3, hide_on_exit=True):
            split_bundles(
                input_tractogram=tractogram_in,
                input_assignments=save_assignments,
                output_folder=output_bundles_folder,
                weights_in=temp_idx,
                max_open=max_open,
                force=force,
                verbose=1)
            bundles = []
            warning_msg = ''
            for dirpath, _, filenames in os.walk(output_bundles_folder):
                for f in filenames:
                    if f.endswith('.tck') and not f.startswith('unassigned'):
                        filename = os.path.abspath(os.path.join(dirpath, f))
                        bundles.append((filename, os.path.getsize(filename), int(info(filename,verbose=1))))
                    if f.startswith('unassigned') and f.endswith('.tck'):
                        warning_msg = f'{int(info(os.path.abspath(os.path.join(dirpath, f)),verbose=1))} streamlines were not assigned to any bundle'
            # Sort the list of tuples by the file size, which is the second element of each tuple
            bundles.sort(key=lambda x: x[1])
            # Convert the sorted list of tuples into a dictionary
            bundles = {i: bundle for i, bundle in enumerate(bundles)}
            bundles[len(bundles)-1] = (bundles[len(bundles)-1][0], bundles[len(bundles)-1][1], bundles[len(bundles)-1][2])
        if warning_msg != '':
            logger.warning(warning_msg) if log_list is None else log_list.append(warning_msg)

        t1 = time.time()
        logger.info( f'[ {format_time(t1 - t0)} ]' )

        ref_indices = []
        streamlines_cluster = []
        TCK_out_size = 0

        # retreieve total memory available
        mem = psutil.virtual_memory()
        mem_avail = mem.available

        logger.info(f'Clustering')
        logger.subinfo(f'Number of input streamlines: {num_streamlines}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Clustering hreshold: {clust_thr}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Number of points: {n_pts}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Computing workload for parallel clustering', indent_lvl=1, indent_char='*', with_progress=verbose>2)
        chunk_list = []
        try:
            TCK_out = LazyTractogram(tractogram_out, mode='w', header=TCK_in.header)
            with ProgressBar(subinfo=True, disable=verbose < 3):
                while True:
                    if max_bytes>0:
                        if max_bytes > mem_avail:
                            MAX_BYTES = mem_avail//MAX_THREAD
                        else:
                            MAX_BYTES = max_bytes//MAX_THREAD
                    else:
                        MAX_BYTES = int(0.9 * mem_avail)//MAX_THREAD

                    executor = ThreadPoolExecutor(max_workers=MAX_THREAD)
                    t0 = time.time()

                    # compute base size of centroid array
                    base_size = getsizeof(np.zeros((1,1, 1000, 3), dtype=np.float32))

                    # compute chunks
                    while len(bundles.items()) > 0:
                        to_delete = []
                        new_chunk = []
                        new_chunk_num_streamlines = []
                        max_bundle_size = 0
                        for k, bundle in bundles.items():
                            new_chunk_size = len(new_chunk) + 1
                            if bundle[2] > max_bundle_size:
                                max_bundle_size = bundle[2]
                            future_size = new_chunk_size * max_bundle_size * 4 * base_size
                            
                            if future_size < MAX_BYTES:
                                new_chunk.append(bundle[0])
                                new_chunk_num_streamlines.append(bundle[2])
                                to_delete.append(k)
                            else:
                                # bundle too big
                                break
                        # remove from bundles list
                        if len(new_chunk_num_streamlines) == 0:
                            MAX_THREAD -= 1
                            break
                        
                        chunk_list.append([new_chunk, max(new_chunk_num_streamlines)])
                        for k in to_delete:
                            bundles.pop(k)

                    if MAX_THREAD == 0:
                        raise ValueError('Not enough memory to process the data')
                    if len(bundles.items()) == 0:
                        break


            logger.subinfo(f'Parallel bundles clustering', indent_lvl=1, indent_char='*', with_progress=verbose>2)
            with ProgressBar(total=len(chunk_list), disable=verbose < 3, hide_on_exit=True, subinfo=True) as pbar:
                future = [executor.submit(cluster_chunk,
                                        chunk,
                                        num_fibs,
                                        clust_thr,
                                        n_pts=n_pts,
                                        metric=metric) for chunk, num_fibs in chunk_list]
                for i, f in enumerate(as_completed(future)):
                    bundle_new_c, bundle_centr_len, bundle_num_c, idx_clst, fib_clust, bundle_size = f.result()

                    for i_b in range(len(bundle_num_c)):
                        ref_indices.extend(idx_clst[i_b][:bundle_num_c[i_b]].tolist())
                        new_centroids, new_centroids_len = bundle_new_c[i_b], bundle_centr_len[i_b]
                        for i_s in range(bundle_num_c[i_b]):
                            TCK_out.write_streamline(new_centroids[i_s, :new_centroids_len[i_s]], new_centroids_len[i_s] )
                            TCK_out_size += 1
                    sum_size = 0
                    for i_b, s_b in enumerate(bundle_size):
                        streamlines_cluster.extend([fib_clust[i_b][s] + sum_size for s in range(s_b)])
                        sum_size += s_b
                        
                    pbar.update()
                TCK_out.close( write_eof=True, count= TCK_out_size)
            
            ret_clust_idx = np.asarray(streamlines_cluster)
            t1 = time.time()
            logger.subinfo(f'Number of computed centroids: {TCK_out_size}', indent_lvl=1, indent_char='*')
            logger.info( f'[ {format_time(t1 - t0)} ]' )
        except Exception as e:
            logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )
            if os.path.isfile(tractogram_out):
                os.remove(tractogram_out)

        os.remove(temp_idx)
        if not keep_temp_files:
            shutil.rmtree(output_bundles_folder)
            os.remove(save_assignments)
            # remove temp_folder if different from current
            if tmp_dir_is_created:
                shutil.rmtree(temp_folder)

        if weights_in is not None:
            w = np.loadtxt(weights_in)
            if weights_metric == 'sum':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.sum(w[fib_indices])
    
    else:
        logger.info(f'Clustering')
        logger.subinfo(f'Number of input streamlines: {num_streamlines}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Clustering threshold: {clust_thr}', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Number of points: {n_pts}', indent_lvl=1, indent_char='*')
        t0 = time.time()

        ref_indices = []
        streamlines_cluster = []

        hash_superset = np.empty( num_streamlines, dtype=int)

        for i in range(num_streamlines):
            TCK_in._read_streamline()
            hash_superset[i] = hash(np.array(TCK_in.streamline[:TCK_in.n_pts]).tobytes())
        TCK_in.close()

        clust_idx, set_centroids = cluster(tractogram_in,
                                            metric=metric,
                                            threshold=clust_thr,
                                            n_pts=n_pts,
                                            verbose=verbose
                                            )

        ret_clust_idx = np.asarray(clust_idx)
        centr_len = np.zeros(set_centroids.shape[0], dtype=np.intc)
        new_c = closest_streamline(tractogram_in, set_centroids, clust_idx, n_pts, set_centroids.shape[0], centr_len)
        
        TCK_out = LazyTractogram(tractogram_out, mode='w', header=TCK_in.header)
        TCK_out_size = 0

        for i, n_c in enumerate(new_c):
            hash_val = hash(np.array(n_c[:centr_len[i]]).tobytes())
            ref_indices.append( np.flatnonzero(hash_superset == hash_val)[0] )
            TCK_out.write_streamline(n_c[:centr_len[i]], centr_len[i] )
            TCK_out_size += 1
        TCK_out.close( write_eof=True, count= TCK_out_size)

        t1 = time.time()
        logger.subinfo(f"Number of computed centroids: {TCK_out_size}", indent_char='*', indent_lvl=1)
        logger.info( f'[ {format_time(t1 - t0)} ]' )
        
        if not keep_temp_files:
            # remove temp_folder if different from current
            if tmp_dir_is_created:
                shutil.rmtree(temp_folder)  

        if weights_in is not None:
            w = np.loadtxt(weights_in)
            if weights_metric == 'sum':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.sum(w[fib_indices])
            elif weights_metric == 'mean':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.mean(w[fib_indices])
            elif weights_metric == 'min':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.min(w[fib_indices])
            elif weights_metric == 'max':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.max(w[fib_indices])
            elif weights_metric == 'median':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.median(w[fib_indices])

            if weights_out.endswith('.txt'):
                np.savetxt(weights_out, cluster_fibs)
            else:
                np.save(weights_out, cluster_fibs, allow_pickle=False)

    if TCK_in is not None:
        TCK_in.close()


    return ref_indices, ret_clust_idx