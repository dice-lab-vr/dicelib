# cython: boundscheck=False, wraparound=False, profile=False, language_level=3
from dicelib.connectivity import _assign
from dicelib.tractogram import info, split
from dicelib.streamline import cumulative_lengths, set_number_of_points
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
from libcpp cimport bool

logger = setup_logger('clustering')


cdef float[:,::1] extract_ending_pts(float[:,::1] fib_in, float[:,::1] resampled_fib) :
    cdef int nb_pts_in = fib_in.shape[0]
    resampled_fib[0][0] = fib_in[0][0]
    resampled_fib[0][1] = fib_in[0][1]
    resampled_fib[0][2] = fib_in[0][2]
    resampled_fib[1][0] = fib_in[nb_pts_in-1][0]
    resampled_fib[1][1] = fib_in[nb_pts_in-1][1]
    resampled_fib[1][2] = fib_in[nb_pts_in-1][2]

    return resampled_fib


cdef (int, int) compute_dist_mean(float[:,::1] fib_in, float[:,:,::1] target, float thr,
                            float d1_x, float d1_y, float d1_z, int num_c, int num_pt) noexcept nogil:
    """Compute the distance between a fiber and a set of centroids"""
    cdef float meandist_pt   = 0
    cdef float meandist_pt_d = 0
    cdef float meandist_pt_i = 0
    cdef float meandist_fib = 3000000000
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
    cdef float maxdist_fib  = 3000000000
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


cpdef cluster(filename_in: str, metric: str="EDavg", threshold: float=4.0, n_pts: int=12,
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
    cdef bool metric_mean = metric == 'EDavg'
    cdef float[:,::1] resampled_fib = np.zeros((nb_pts,3), dtype=np.float32)
    cdef float[:,:,::1] set_centroids = np.zeros((n_streamlines,nb_pts,3), dtype=np.float32)
    cdef float[:,::1] s0 = np.empty( (n_pts, 3), dtype=np.float32 )
    cdef float[:] lengths = np.zeros(3000, dtype=np.float32)
    TCK_in.read_streamline()
    cdef size_t pp = 0

    if TCK_in.n_pts == nb_pts: # no need to resample
        for pp in xrange(nb_pts): # copy streamline
            s0[pp][0] = TCK_in.streamline[pp][0]
            s0[pp][1] = TCK_in.streamline[pp][1]
            s0[pp][2] = TCK_in.streamline[pp][2]
    else:
        set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], nb_pts, s0, lengths )

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
            TCK_in.read_streamline()
            if TCK_in.n_pts == nb_pts: # no need to resample
                for pp in xrange(nb_pts): # copy streamline
                    streamline_in[pp][0] = TCK_in.streamline[pp][0]
                    streamline_in[pp][1] = TCK_in.streamline[pp][1]
                    streamline_in[pp][2] = TCK_in.streamline[pp][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], nb_pts, streamline_in[:], lengths)

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
    cdef float[:] lengths = np.zeros(3000, dtype=np.float32)
    cdef size_t p = 0


    with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=True) as pbar:
        for i_f in xrange(n_streamlines):
            TCK_in.read_streamline()
            c_i = clust_idx[i_f]
            if TCK_in.n_pts == num_pt: # no need to resample
                for p in xrange(num_pt): # copy streamline
                    fib_in[p][0] = TCK_in.streamline[p][0]
                    fib_in[p][1] = TCK_in.streamline[p][1]
                    fib_in[p][2] = TCK_in.streamline[p][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], num_pt, fib_in[:], lengths )
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


cpdef cluster_chunk(filenames: list[str], num_fibs: int, threshold: float=10.0, n_pts: int=10, metric: str="EDavg"):
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
    cdef float[:] lengths = np.zeros(3000, dtype=np.float32)

    for i, filename in enumerate(filenames):
        TCK_in = LazyTractogram( filename, mode='r', max_points=1000 )
        idx = np.load(f'{filename[:len(filename)-4]}.npy').astype(np.intc)
        idx_cl[i, :idx.shape[0]] = idx
        n_streamlines[i] = int(TCK_in.header['count'])
        header_params[i] = int(TCK_in.header['file'][2:])
        TCK_in.read_streamline()
        if TCK_in.n_pts == n_pts: # no need to resample
            for pp in xrange(n_pts): # copy streamline
                set_centroids[i, 0, pp, 0] = TCK_in.streamline[pp][0]
                set_centroids[i, 0, pp, 1] = TCK_in.streamline[pp][1]
                set_centroids[i, 0, pp, 2] = TCK_in.streamline[pp][2]
        else:
            set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], n_pts, set_centroids[i, 0], lengths )
        TCK_in.close()


    in_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines)), 1000, 3), dtype=np.float32)

    cdef float[:,:,:,::1] in_streamlines_view = in_streamlines
    cdef int [:,::1] len_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines))), dtype=np.int32)
    cdef float[:,:,:,::1] resampled_streamlines = np.zeros((len(filenames), int(np.max(n_streamlines)), n_pts, 3), dtype=np.float32)

    for i, filename in enumerate(filenames):
        TCK_in = LazyTractogram( filename, mode='r', max_points=1000 )
        for st in range(n_streamlines[i]):
            TCK_in.read_streamline()
            in_streamlines[i][st][:TCK_in.n_pts] = TCK_in.streamline[:TCK_in.n_pts]
            len_streamlines[i][st] = TCK_in.n_pts
            if TCK_in.n_pts == n_pts: # no need to resample
                for pp in xrange(n_pts): # copy streamline
                    resampled_streamlines[i, st, pp, 0] = TCK_in.streamline[pp][0]
                    resampled_streamlines[i, st, pp, 1] = TCK_in.streamline[pp][1]
                    resampled_streamlines[i, st, pp, 2] = TCK_in.streamline[pp][2]
            else:
                set_number_of_points( TCK_in.streamline[:TCK_in.n_pts], n_pts, resampled_streamlines[i, st], lengths)
        TCK_in.close()

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
    cdef bool metric_mean = metric == 'EDavg'

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


    return clst_streamlines, centr_len, new_c, idx_cl_return, clust_idx, bundle_n_streamlines, idx_cl


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


def run_clustering( tractogram_filename: str, thr: float, out_tractogram_filename: str, metric: str="EDavg", n_pts: int=12,
                    atlas: str=None, atlas_thr: float=2.0, save_clust_idx: bool=False,
                    scalars_filename: str=None, out_scalars_filename: str=None, scalars_stat: str="sum",
                    tmp_folder: str='tmp_cluster', keep_tmp: bool=False,
                    n_threads: int=None, max_open: int=None, max_bytes: int=0, force: bool=False, verbose: int=3, log_list=None):
    """Cluster streamlines in a tractogram based on a given distance metric.

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    thr : float
        Threshold to use when computing distances between the streamlines.
    out_tractogram_filename : str
        Path to the tractogram (.tck) that will contain the clustered streamlines.
    metric : {'EDavg', 'EDmax'}, default='EDavg'
        Metric to use for computing distances between streamlines:
        - 'EDavg' = average pointwise Euclidean Distance (i.e., streamlines with
          average Euclidean distance smaller than 'thr' will be clustered together);
        - 'EDmax' = max pointwise Euclidean Distance (i.e., streamlines with
          max Euclidean distance smaller than 'thr' will be clustered together).
    n_pts : int, default=12
        Number of points to resample the streamlines before clustering.
        NB: this clustering algorithm requires all streamlines to have the same number of points.
    atlas : str, optional
        Path to the image (.nii, .nii.gz) containing the atlas used to split the streamlines
        into bundles and cluster each of them in parallel; if not specified, the clustering
        will be performed sequentially on the whole tractogram (may be slow).
    atlas_thr : float, default=2.0
        Distance [in voxels] to consider in the radial search when computing the assignments
        to be used fo splitting in bundles using the atlas. If no label is found within this radius,
        the corresponding streamline is not taken into account for clustering.
    save_clust_idx : bool, default=False
        Save the indices of the cluster to which each input streamline belongs.
    scalars_filename : str, optional
        Path to the file (.txt, .npy) containing one scalar for each input streamline.
    out_scalars_filename : str, optional
        Path to the file (.txt, .npy) that will contain the scalars of the centroids.
    scalars_stat : {'sum', 'mean', 'median', 'min', 'max'}, default='sum'
        Summary statistic to use for computing the scalar of a centroid from the streamlines
        that were assigned to it.
    tmp_folder : str, default='tmp_cluster'
        Path to the temporary folder used to store the intermediate files.
    keep_tmp : boolean, default=False
        Keep the temporary folder.
    n_threads : int, optional
        Number of threads to use in parallel operations; if not specified,
        all available threads will be used.
    max_open : int, optional
        Maximum number of files opened at the same time that can be used when
        splitting the input streamlines into bundles for parallel clustering.
    max_bytes : int, default=0
        !!! MISSING DOCUMENTATION !!!
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    t0 = time.time()
    set_verbose('clustering', verbose)
    logger.info(f'Clustering')

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck']),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext=['.tck'])
    ]
    if atlas is not None:
        files.append(File(name='atlas', type_='input', path=atlas, ext=['.nii', '.nii.gz']))
    tmp_folder = tmp_folder if tmp_folder is not None else os.path.join(os.getcwd(), 'tmp')
    dirs = [
        Dir(name='tmp_folder', path=tmp_folder)
    ]
    if scalars_filename is not None:
        files.append(File(name='scalars_filename', type_='input', path=scalars_filename, ext=['.txt', '.npy']))
    if out_scalars_filename is not None:
        files.append(File(name='out_scalars_filename', type_='output', path=out_scalars_filename, ext=['.txt', '.npy']))
    nums = [
        Num(name='thr', value=thr, min_=0.0, include_min=False),
        Num(name='atlas_thr', value=atlas_thr, min_=0.0, include_min=True),
        Num(name='n_pts', value=n_pts, min_=2)
    ]
    if n_threads is not None:
        nums.append(Num(name='n_threads', value=n_threads, min_=1))

    if scalars_stat not in ['min', 'max', 'median', 'sum', 'mean']:
        logger.error(f'Option {scalars_stat} not valid, please choose between min, max, median, mean or sum')
    check_params(files=files, dirs=dirs, nums=nums, force=force)

    tmp_dir_is_created = False
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        tmp_dir_is_created = True

    # other checks
    if metric not in ['EDavg', 'EDmax']:
        logger.error(f'Invalid metric, must be \'EDavg\' or \'EDmax\'')

    def compute_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    if scalars_filename:
        if scalars_filename.endswith('.txt'):
            w = np.loadtxt(scalars_filename).astype(np.float64)
        else:
            w = np.load(scalars_filename, allow_pickle=False).astype(np.float64)

    if n_threads:
        MAX_THREAD = n_threads
    else:
        MAX_THREAD = os.cpu_count()

    TCK_in = LazyTractogram(tractogram_filename, mode='r')
    num_streamlines = int(TCK_in.header["count"])

    logger.subinfo(f'Number of streamlines: {num_streamlines}', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Clustering metric: "{metric}"', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Clustering threshold: {thr}', indent_lvl=1, indent_char='*')
    logger.subinfo(f'Points per streamline: {n_pts}', indent_lvl=1, indent_char='*')
    if scalars_filename is not None:
        logger.debug( f'Streamline scalars filename: "{scalars_filename}"' )
        logger.subinfo(f'Statistic to summarize centroid scalars: "{scalars_stat}"', indent_lvl=1, indent_char='*')
        if out_scalars_filename is not None:
            logger.debug( f'Streamline scalars filename: "{scalars_filename}"' )
        else:
            logger.warning( f'Streamline scalars passed as input, but not as output' )
    elif out_scalars_filename is not None:
        logger.warning( f'Streamline scalars passed as output, but not as intput' )

    if atlas:
        chunk_size = int(num_streamlines/MAX_THREAD)
        chunk_groups = [e for e in compute_chunks( np.arange(num_streamlines),chunk_size)]

        # check if save_assignments is None
        save_assignments = os.path.join(tmp_folder, f'{os.path.basename(tractogram_filename)[:-4]}_assignments.txt')
        temp_idx_arr = np.arange(num_streamlines)
        temp_idx = os.path.join(tmp_folder, 'streamline_idx.npy')
        np.save( temp_idx, temp_idx_arr )

        chunks_asgn = []

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
                        _assign,
                        tractogram_filename,
                        pbar_array,
                        i,
                        start_chunk=int(chunk_groups[i][0]),
                        end_chunk=int(chunk_groups[i][len(chunk_groups[i])-1]+1),
                        gm_map_data=atlas_data,
                        gm_map_img=atlas_img,
                        threshold=atlas_thr ) for i in range(len(chunk_groups))]
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

        output_bundles_folder = os.path.join(tmp_folder, 'bundles')
        logger.info('Splitting the bundles into separate files')
        with ProgressBar(disable=verbose<3, hide_on_exit=True):
            split(
                tractogram_filename=tractogram_filename,
                assignments_filename=save_assignments,
                out_folder=output_bundles_folder,
                scalars_filename=temp_idx,
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
        w_out = []
        TCK_out_size = 0

        # retreieve total memory available
        mem = psutil.virtual_memory()
        mem_avail = mem.available

        logger.subinfo(f'Computing workload for parallel clustering', indent_lvl=1, indent_char='*', with_progress=verbose>2)
        chunk_list = []
        try:
            TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
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


            tot_centroids = 0
            idx_centroid_per_streamline = np.full(num_streamlines, np.nan)
            logger.subinfo(f'Parallel bundles clustering', indent_lvl=1, indent_char='*', with_progress=verbose>2)
            with ProgressBar(total=len(chunk_list), disable=verbose < 3, hide_on_exit=True, subinfo=True) as pbar:
                future = [executor.submit(cluster_chunk,
                                        chunk,
                                        num_fibs,
                                        thr,
                                        n_pts=n_pts,
                                        metric=metric) for chunk, num_fibs in chunk_list]
                for i, f in enumerate(as_completed(future)):
                    bundle_new_c, bundle_centr_len, bundle_num_c, idx_clst, fib_clust, bundle_size, idx_cl = f.result()

                    for i_b in range(len(bundle_num_c)):
                        ref_indices.extend(idx_clst[i_b][:bundle_num_c[i_b]].tolist())
                        new_centroids, new_centroids_len = bundle_new_c[i_b], bundle_centr_len[i_b]
                        for i_s in range(bundle_num_c[i_b]):
                            TCK_out.write_streamline(new_centroids[i_s, :new_centroids_len[i_s]], new_centroids_len[i_s] )
                            TCK_out_size += 1

                    for i_b, s_b in enumerate(bundle_size):
                        streamlines_cluster = [fib_clust[i_b][s] for s in range(s_b)]
                        streamline_indices = [idx_cl[i_b][s] for s in range(s_b)]
                        # save idx of centroid per input streamline
                        idx_centroid_per_streamline[streamline_indices] = np.array(streamlines_cluster) + tot_centroids
                        tot_centroids += np.array(streamlines_cluster).max() + 1
                        # compute weights
                        if scalars_filename is not None:
                            if scalars_stat == 'sum':
                                clusters_v = np.unique(streamlines_cluster)
                                for c in clusters_v:
                                    fib_indices = np.where(streamlines_cluster == c)[0]
                                    tmp_i = [streamline_indices[ii] for ii in fib_indices]
                                    tmp_w = w[tmp_i]
                                    w_out.append(np.sum(tmp_w))
                            elif scalars_stat == 'mean':
                                clusters_v = np.unique(streamlines_cluster)
                                for c in clusters_v:
                                    fib_indices = np.where(streamlines_cluster == c)[0]
                                    tmp_i = [streamline_indices[ii] for ii in fib_indices]
                                    tmp_w = w[tmp_i]
                                    w_out.append(np.mean(tmp_w))
                            elif scalars_stat == 'min':
                                clusters_v = np.unique(streamlines_cluster)
                                for c in clusters_v:
                                    fib_indices = np.where(streamlines_cluster == c)[0]
                                    tmp_i = [streamline_indices[ii] for ii in fib_indices]
                                    tmp_w = w[tmp_i]
                                    w_out.append(np.min(tmp_w))
                            elif scalars_stat == 'max':
                                clusters_v = np.unique(streamlines_cluster)
                                for c in clusters_v:
                                    fib_indices = np.where(streamlines_cluster == c)[0]
                                    tmp_i = [streamline_indices[ii] for ii in fib_indices]
                                    tmp_w = w[tmp_i]
                                    w_out.append(np.max(tmp_w))
                            elif scalars_stat == 'median':
                                clusters_v = np.unique(streamlines_cluster)
                                for c in clusters_v:
                                    fib_indices = np.where(streamlines_cluster == c)[0]
                                    tmp_i = [streamline_indices[ii] for ii in fib_indices]
                                    tmp_w = w[tmp_i]
                                    w_out.append(np.median(tmp_w))

                    pbar.update()
                TCK_out.close( write_eof=True, count= TCK_out_size)

            if out_scalars_filename is not None:
                w_out = np.array(w_out)
                if out_scalars_filename.endswith('.txt'):
                    np.savetxt(out_scalars_filename, w_out)
                else:
                    np.save(out_scalars_filename, w_out, allow_pickle=False)

            ret_clust_idx = idx_centroid_per_streamline

        except Exception as e:
            logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )
            if os.path.isfile(out_tractogram_filename):
                os.remove(out_tractogram_filename)

        os.remove(temp_idx)
        if not keep_tmp:
            shutil.rmtree(output_bundles_folder)
            os.remove(save_assignments)
            # remove tmp_folder if different from current
            if tmp_dir_is_created:
                shutil.rmtree(tmp_folder)


    else:
        ref_indices = []
        streamlines_cluster = []
        hash_superset = np.empty(num_streamlines, dtype=np.int64)
        for i in range(num_streamlines):
            TCK_in.read_streamline()
            hash_superset[i] = hash(np.array(TCK_in.streamline[:TCK_in.n_pts]).tobytes())
        TCK_in.close()

        clust_idx, set_centroids = cluster(
            tractogram_filename,
            metric=metric,
            threshold=thr,
            n_pts=n_pts,
            verbose=verbose
        )

        ret_clust_idx = np.asarray(clust_idx)
        centr_len = np.zeros(set_centroids.shape[0], dtype=np.intc)
        new_c = closest_streamline(tractogram_filename, set_centroids, clust_idx, n_pts, set_centroids.shape[0], centr_len)

        TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
        TCK_out_size = 0

        for i, n_c in enumerate(new_c):
            hash_val = hash(np.array(n_c[:centr_len[i]]).tobytes())
            ref_indices.append( np.flatnonzero(hash_superset == hash_val)[0] )
            TCK_out.write_streamline(n_c[:centr_len[i]], centr_len[i] )
            TCK_out_size += 1
        TCK_out.close( write_eof=True, count= TCK_out_size)

        if not keep_tmp:
            # remove tmp_folder if different from current
            if tmp_dir_is_created:
                shutil.rmtree(tmp_folder)

        if scalars_filename is not None:
            #w = np.loadtxt(scalars_filename)
            if scalars_stat == 'sum':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.sum(w[fib_indices])
            elif scalars_stat == 'mean':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.mean(w[fib_indices])
            elif scalars_stat == 'min':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.min(w[fib_indices])
            elif scalars_stat == 'max':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.max(w[fib_indices])
            elif scalars_stat == 'median':
                cluster_fibs = np.zeros(len(ref_indices), dtype=np.float32)
                for i in range(len(ref_indices)):
                    fib_indices = np.where(ret_clust_idx == i)[0]
                    cluster_fibs[i] = np.median(w[fib_indices])

            if out_scalars_filename is not None:
                if out_scalars_filename.endswith('.txt'):
                    np.savetxt(out_scalars_filename, cluster_fibs)
                else:
                    np.save(out_scalars_filename, cluster_fibs, allow_pickle=False)

    if TCK_in is not None:
        TCK_in.close()

    if save_clust_idx:
        np.savetxt(f'{out_tractogram_filename[:len(out_tractogram_filename)-4]}_clust_idx.txt', ret_clust_idx, fmt='%d')

    t1 = time.time()
    logger.subinfo(f"Number of output centroids: {TCK_out_size}", indent_char='*', indent_lvl=1)
    logger.info( f'[ {format_time(t1 - t0)} ]' )

    return ref_indices, ret_clust_idx


cpdef closest_centroid_pt(float[:,::1] centroid, float[:,::1] streamline, float[:] streamline_values, int num_pt):

    cdef float dist_d = 0
    cdef float dist_f = 0
    cdef float dist_min = 1e6
    cdef float d_x = 0
    cdef float d_y = 0
    cdef float d_z = 0
    cdef size_t  i = 0
    cdef size_t  j = 0
    cdef float[:] proj_values = np.zeros(num_pt, dtype=np.float32)

    for i in xrange(num_pt):

        dist_d = 0
        dist_f = 0
        dist_min = 1e6
        for j in xrange(num_pt):
            # direct
            d_x = (centroid[i][0] - streamline[j][0])**2
            d_y = (centroid[i][1] - streamline[j][1])**2
            d_z = (centroid[i][2] - streamline[j][2])**2
            dist_d = sqrt(d_x + d_y + d_z)

            # flipped
            d_x = (centroid[i][0] - streamline[num_pt-j-1][0])**2
            d_y = (centroid[i][1] - streamline[num_pt-j-1][1])**2
            d_z = (centroid[i][2] - streamline[num_pt-j-1][2])**2
            dist_f = sqrt(d_x + d_y + d_z)

            if dist_d < dist_f:
                if dist_d < dist_min:
                    dist_min = dist_d
                    proj_values[i] = streamline_values[j]
            else:
                if dist_f < dist_min:
                    dist_min = dist_f
                    proj_values[i] = streamline_values[num_pt-j-1]

    return proj_values


cpdef project_values_on_centroid(filename_tractogram: str, float[:,:] streamline_vals, thr: float=20.0 ):

    cdef LazyTractogram TCK_in = LazyTractogram( filename_tractogram, mode='r', max_points=1000 )
    cdef float[:] lengths = np.zeros(3000, dtype=np.float32)
    cdef float[:,::1] centroid_resampled = np.empty( (256, 3), dtype=np.float32 )
    cdef float[:,::1] streamline_resampled = np.empty( (256, 3), dtype=np.float32 )
    cdef float[:] proj_values = np.zeros(256, dtype=np.float32)
    cdef float[:] final_values = np.zeros(256, dtype=np.float32)
    cdef float[:,:] streamline_values = streamline_vals
    cdef int num_str = int( TCK_in.header['count'] )
    cdef size_t i = 0
    cdef size_t j = 0

    _, centroid = cluster(filename_tractogram, threshold=thr, n_pts=12)
    set_number_of_points( centroid[0], 256, centroid_resampled, lengths)

    for i in xrange(num_str):
        proj_values[:] = 0
        TCK_in.read_streamline()
        set_number_of_points(TCK_in.streamline[:TCK_in.n_pts], 256, streamline_resampled, lengths)
        proj_values = closest_centroid_pt(centroid_resampled, streamline_resampled, streamline_values[i], 256)
        for j in xrange(256):
            final_values[j] += proj_values[j]

    TCK_in.close()
    return np.asarray(final_values)
