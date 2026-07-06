# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, language_level=3
from dicelib.streamline cimport set_number_of_points_f64
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, Dir, File, Num, format_time
import os
from time import time
import numpy as np
cimport numpy as np
from dicelib.tractogram cimport LazyTractogram
from libc.math cimport sqrtf, fabs
from scipy.fft import dct

logger = setup_logger('clustering')


cdef class DistanceMetric:
    """Base class to define new distance metrics; new metrics should inherit from this class.
    All the methods need to be overloaded to account for the specific needs of the metric.
    Each method will then be called by a dispatcher when needed.

    Attributes
    ----------
    n_pts : integer
        The number of coordinates that each streamline must have to compute the distance.
    """
    # cdef:
    #     int n_pts

    def __init__( self, int n_pts ) :
        self.n_pts = n_pts
        return

    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil:
        raise NotImplementedError("Subclasses must implement this function")

    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_label, int* out_flipped) nogil:
        raise NotImplementedError("Subclasses must implement this function")

    def __dealloc__(self):
        pass


cdef class AverageEuclideanDistance(DistanceMetric):
    """Average Euclidean Distance (AED) between streamlines"""

    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil:
        """Calculate the distance between two streamlines"""
        cdef:
            size_t j, k
            double dx, dy, dz
            double dist_direct = 0, dist_flipped = 0

        for j in range(self.n_pts):
            dx = f[j,0] - g[j,0]
            dy = f[j,1] - g[j,1]
            dz = f[j,2] - g[j,2]
            dist_direct += sqrtf( dx*dx + dy*dy + dz*dz )

            k = self.n_pts-j-1
            dx = f[k,0] - g[j,0]
            dy = f[k,1] - g[j,1]
            dz = f[k,2] - g[j,2]
            dist_flipped += sqrtf( dx*dx + dy*dy + dz*dz )

        # Only update if we actually found a new minimum
        if dist_direct <= dist_flipped:
            out_flipped[0] = 0
            return dist_direct
        else:
            out_flipped[0] = 1
            return dist_flipped


    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_label, int* out_flipped) nogil:
        """Calculate the distance between a streamline and a set of centroids"""
        cdef:
            size_t i, j, k
            double dx, dy, dz
            double dist_direct, dist_flipped
            double dist_min_all = self.n_pts * thr
            int label = n_clusters, flipped = 0

        for i in range(n_clusters):
            dist_direct = 0
            dist_flipped = 0
            for j in range(self.n_pts):
                dx = streamline[j,0] - <double>centroids[i,j,0]
                dy = streamline[j,1] - <double>centroids[i,j,1]
                dz = streamline[j,2] - <double>centroids[i,j,2]
                dist_direct += sqrtf( dx*dx + dy*dy + dz*dz )

                k = self.n_pts-j-1
                dx = streamline[k,0] - <double>centroids[i,j,0]
                dy = streamline[k,1] - <double>centroids[i,j,1]
                dz = streamline[k,2] - <double>centroids[i,j,2]
                dist_flipped += sqrtf( dx*dx + dy*dy + dz*dz )

                # if both direct and flipped distances are already worse
                # than best found distance, no need to continue computing
                if j % 4 == 0:
                    if dist_direct >= dist_min_all and dist_flipped >= dist_min_all:
                        break

            # Only update if we actually found a new minimum
            if dist_direct <= dist_flipped:
                if dist_direct < dist_min_all:
                    dist_min_all = dist_direct
                    flipped = 0
                    label = i
            else:
                if dist_flipped < dist_min_all:
                    dist_min_all = dist_flipped
                    flipped = 1
                    label = i

        out_label[0] = label
        out_flipped[0] = flipped
        return dist_min_all


cdef class AverageSquaredEuclideanDistance(DistanceMetric):
    """Average Squared Euclidean Distance (ASED) between streamlines."""

    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil:
        """Calculate the distance between two streamlines"""
        cdef:
            size_t j, k
            double dist_direct=0, dist_flipped=0
        for j in range(self.n_pts):
            dist_direct += (f[j,0]-g[j,0])*(f[j,0]-g[j,0]) + (f[j,1]-g[j,1])*(f[j,1]-g[j,1]) + (f[j,2]-g[j,2])*(f[j,2]-g[j,2])
            k = self.n_pts-j-1
            dist_flipped += (f[k,0]-g[j,0])*(f[k,0]-g[j,0]) + (f[k,1]-g[j,1])*(f[k,1]-g[j,1]) + (f[k,2]-g[j,2])*(f[k,2]-g[j,2])
        if dist_direct <= dist_flipped:
            out_flipped[0] = 0
            return dist_direct/self.n_pts
        else:
            out_flipped[0] = 1
            return dist_flipped/self.n_pts


    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_label, int* out_flipped) nogil:
        """Calculate the distance between a streamline and a set of centroids"""
        cdef:
            size_t i, j, k
            double dx, dy, dz
            double dist_direct, dist_flipped
            double dist_min_all = self.n_pts * thr
            int label = n_clusters, flipped = 0

        for i in range(n_clusters):
            dist_direct = 0
            dist_flipped = 0
            for j in range(self.n_pts):
                dx = streamline[j,0] - <double>centroids[i,j,0]
                dy = streamline[j,1] - <double>centroids[i,j,1]
                dz = streamline[j,2] - <double>centroids[i,j,2]
                dist_direct += dx*dx + dy*dy + dz*dz
                k = self.n_pts-j-1
                dx = streamline[k,0] - <double>centroids[i,j,0]
                dy = streamline[k,1] - <double>centroids[i,j,1]
                dz = streamline[k,2] - <double>centroids[i,j,2]
                dist_flipped += dx*dx + dy*dy + dz*dz
                if dist_direct >= dist_min_all and dist_flipped >= dist_min_all:
                    break

            # only update if a new minimum is found
            if dist_direct <= dist_flipped:
                if dist_direct < dist_min_all:
                    dist_min_all = dist_direct
                    flipped = 0
                    label = i
            else:
                if dist_flipped < dist_min_all:
                    dist_min_all = dist_flipped
                    flipped = 1
                    label = i

        out_label[0] = label
        out_flipped[0] = flipped
        return dist_min_all


cdef class AverageSquaredEuclideanDistanceDCT(DistanceMetric):
    """Average Squared Euclidean Distance (ASED) between streamlines in DCT space."""
    # cdef:
    #     int n_dct

    def __init__( self, int n_pts, int n_dct ) :
        if n_dct%2==1:
            raise ValueError( f'[AverageSquaredEuclideanDistanceDCT] n_dct must be even' )
        self.n_pts = n_pts
        self.n_dct = n_dct
        return

    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil:
        """Calculate the distance between two streamlines"""
        cdef:
            size_t j
            double dist_direct, dist_flipped
            double tmp1 = 0, tmp2 = 0, tmp3 = 0
        for j in range(0, self.n_dct, 2):
            tmp1 += (f[j,0]-g[j,0])*(f[j,0]-g[j,0]) + (f[j,1]-g[j,1])*(f[j,1]-g[j,1]) + (f[j,2]-g[j,2])*(f[j,2]-g[j,2])
        for j in range(1, self.n_dct, 2):
            tmp2 += (f[j,0]-g[j,0])*(f[j,0]-g[j,0]) + (f[j,1]-g[j,1])*(f[j,1]-g[j,1]) + (f[j,2]-g[j,2])*(f[j,2]-g[j,2])
            tmp3 += (f[j,0]+g[j,0])*(f[j,0]+g[j,0]) + (f[j,1]+g[j,1])*(f[j,1]+g[j,1]) + (f[j,2]+g[j,2])*(f[j,2]+g[j,2])
        dist_direct  = tmp1+tmp2
        dist_flipped = tmp1+tmp3
        if dist_direct <= dist_flipped:
            out_flipped[0] = 0
            return dist_direct/self.n_pts #FIXME: use the right number of coeffs
        else:
            out_flipped[0] = 1
            return dist_flipped/self.n_pts #FIXME: use the right number of coeffs


    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_label, int* out_flipped) nogil:
        """Calculate the distance between a streamline and a set of centroids"""
        cdef:
            size_t i, j, k
            double dx, dy, dz
            double tmp1
            double dist_direct, dist_flipped
            double dist_min_all = self.n_pts * thr #FIXME: use the right number of coeffs
            int label = n_clusters, flipped = 0

        for i in range(n_clusters):
            dist_direct = 0
            dist_flipped = 0
            for j in range(0, self.n_dct, 2):
                # even = [0, 2, 4, ...]
                dx = streamline[j,0]-<double>centroids[i,j,0]
                dy = streamline[j,1]-<double>centroids[i,j,1]
                dz = streamline[j,2]-<double>centroids[i,j,2]
                tmp1 = dx*dx + dy*dy + dz*dz
                dist_direct  += tmp1
                dist_flipped += tmp1
                if dist_direct >= dist_min_all and dist_flipped >= dist_min_all:
                    break

                # odd = [1, 3, 5, ...]
                k = j+1
                dx  = streamline[k,0]-<double>centroids[i,k,0]
                dy  = streamline[k,1]-<double>centroids[i,k,1]
                dz  = streamline[k,2]-<double>centroids[i,k,2]
                dist_direct  += dx*dx + dy*dy + dz*dz
                dx = streamline[k,0]+<double>centroids[i,k,0]
                dy = streamline[k,1]+<double>centroids[i,k,1]
                dz = streamline[k,2]+<double>centroids[i,k,2]
                dist_flipped += dx*dx + dy*dy + dz*dz
                if dist_direct >= dist_min_all and dist_flipped >= dist_min_all:
                    break

            # only update if a new minimum is found
            if dist_direct <= dist_flipped:
                if dist_direct < dist_min_all:
                    dist_min_all = dist_direct
                    flipped = 0
                    label = i
            else:
                if dist_flipped < dist_min_all:
                    dist_min_all = dist_flipped
                    flipped = 1
                    label = i

        out_label[0] = label
        out_flipped[0] = flipped
        return dist_min_all


cpdef cluster( str tractogram_filename, float thr, str out_tractogram_filename,
               str metric="AED", int n_points=12, int n_dct_coeffs=12,
               bool ret_centroids=False, str out_labels_filename=None,
               int chunk_size=10000, bool force=False, int verbose=3 ):
    """Cluster streamlines in a tractogram with QuickBundles [1].

    Streamlines with distance smaller than 'thr' will be clustered together.

    References:
    [1] https://doi.org/10.3389/fnins.2012.00175

    Parameters
    ----------
    tractogram_filename : str
        Path to the tractogram (.tck) containing the streamlines to process.
    thr : float
        Threshold to use when computing distances between the streamlines.
    out_tractogram_filename : str
        Path to the tractogram (.tck) that will contain the clustered streamlines.
    metric : {'AED','ASED','ASEDdct'}, default='AED'
        Metric to use for computing distances between streamlines:
        - AED     = Average Euclidean Distance
        - ASED    = Average Squared Euclidean Distance
        - ASEDdct = Average Squared Euclidean Distance in DCT space
    n_points : int, default=12
        Streamlines are resampled to this number of points before clustering,
        since the algorithm requires streamlines to have the same number of points.
    n_dct_coeffs : int, default=12
        If 'ASEDdct' is used as distance, a streamline is first resampled to n_points
        points, then the Discrete Cosine Transform (DCT) is computed and, finally,
        the distance is calculated using only the first n_dct_coeffs coefficients.
    ret_centroids : boolean, default=False
        Whether to return the centroids (i.e. mean streamline is a cluster) or medoids
        (i.e. closest streamline to a centroid) as cluster representatives.
    out_labels_filename : str, optional
        Path to the scalar file (.txt, .npy) that will contain the index of
        the cluster each input streamline belongs to.
    chunk_size : int, default=10000
        The number of centroids to keep in memory is dynamically incremented,
        when needed, by chunks of this value.
    force : boolean, default=False
        Force overwriting of the output files.
    verbose : int, default=3
        What information to print, must be in [0...4] as defined in ui.set_verbose().
    """
    cdef:
        int n_streamlines, n_clusters, n_pts = n_points, n_dct = n_dct_coeffs
        float [:,:,::1] centroids
        int [::1] medoid_idx
        int [:] cluster_size
        double [:,::1] streamline
        double [:,::1] centroid
        double [:,::1] streamline_res
        double [:,::1] dct_M
        int [:] labels
        int c_idx, is_flipped
        size_t i, j, k
        double n1, n2, d
        double [:] closest_streamline_distance
        DistanceMetric distance
        LazyTractogram TCK_in = None, TCK_out = None
        float[:] lengths = np.empty(3000, dtype=np.float32)

    t0 = time()
    set_verbose('clustering', verbose)
    logger.info('Clustering tractogram')

    if metric not in ['AED', 'ASED', 'ASEDdct']:
        logger.error(f"Metric '{metric}' not recognized")
        return

    files = [
        File(name='tractogram_filename', type_='input', path=tractogram_filename, ext=['.tck']),
        File(name='out_tractogram_filename', type_='output', path=out_tractogram_filename, ext=['.tck'])
    ]
    if out_labels_filename is not None:
        files.append(File(name='out_labels_filename', type_='output', path=out_labels_filename, ext=['.txt', '.npy']))
    nums = [
        Num(name='thr', value=thr, min_=0.0, include_min=False),
        Num(name='n_points', value=n_points, min_=2),
        Num(name='chunk_size', value=chunk_size, min_=100)
    ]
    if metric=='ASEDdct':
        nums.append( Num(name='n_dct_coeffs', value=n_dct_coeffs, min_=1, max_=n_points) )
        if n_dct%2==1:
            logger.error( f'n_dct_coeffs must be even' )
    check_params(files=files, nums=nums, force=force)


    try:
        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int(TCK_in.header['count'])
        logger.subinfo(f'Input:', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=2, indent_char='-')
        if metric=='ASEDdct':
            logger.subinfo(f'Distance metric: {metric} (using {n_dct_coeffs} points/streamline)', indent_lvl=2, indent_char='-')
        else:
            logger.subinfo(f'Distance metric: {metric} (using {n_points} points/streamline)', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Distance threshold: {thr}', indent_lvl=2, indent_char='-')
        logger.debug(f'Memory for centroids is dynamically incremented in chunks of {chunk_size} items')
        if n_streamlines == 0:
            return

        # initialize data structures
        logger.subinfo(f"Clustering:", indent_char='*', indent_lvl=1, with_progress=True)
        if metric == 'AED':
            distance = AverageEuclideanDistance( n_pts )
        elif metric == 'ASED':
            distance = AverageSquaredEuclideanDistance( n_pts )
        elif metric == 'ASEDdct':
            if n_dct > n_pts:
                logger.error( f'n_dct must be <= n_pts' )
            distance = AverageSquaredEuclideanDistanceDCT( n_pts, n_dct )
            dct_M = dct( np.eye(n_pts), axis=0, norm="ortho" )[:n_dct,:]

        streamline_res = np.empty((n_pts,3), dtype=np.float64)
        if metric != 'ASEDdct':
            streamline = np.empty((n_pts,3), dtype=np.float64)
            centroid = np.empty((n_pts,3), dtype=np.float64)
            centroids_np = np.empty((chunk_size, n_pts, 3), dtype=np.float32)
            centroids = centroids_np
        else:
            streamline = np.empty((n_dct,3), dtype=np.float64)
            centroid = np.empty((n_dct,3), dtype=np.float64)
            centroids_np = np.empty((chunk_size, n_dct, 3), dtype=np.float32)
            centroids = centroids_np

        cluster_size_np = np.ones(chunk_size, dtype=np.int32)
        cluster_size = cluster_size_np
        labels = np.empty(n_streamlines, dtype=np.int32)

        # Process first streamline
        n_clusters = 1
        labels[0] = 0
        TCK_in.read_streamline()
        set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline_res, n_pts, lengths)
        if metric != 'ASEDdct':
            for j in range(n_pts):
                centroids[0,j,0] = streamline_res[j,0]
                centroids[0,j,1] = streamline_res[j,1]
                centroids[0,j,2] = streamline_res[j,2]
        else:
            tmp = np.dot( dct_M, streamline_res )
            for j in range(n_dct):
                centroids[0,j,0] = tmp[j,0]
                centroids[0,j,1] = tmp[j,1]
                centroids[0,j,2] = tmp[j,2]

        # Process remaining streamlines
        with ProgressBar(total=n_streamlines-1, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
            for i in range(1, n_streamlines):
                TCK_in.read_streamline()
                if metric != 'ASEDdct':
                    set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline, n_pts, lengths)
                else:
                    set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline_res, n_pts, lengths)
                    tmp = np.dot( dct_M, streamline_res )
                    for j in range(n_dct):
                        streamline[j,0] = tmp[j,0]
                        streamline[j,1] = tmp[j,1]
                        streamline[j,2] = tmp[j,2]

                dist_min_all = distance.closest_centroid(streamline, centroids, n_clusters, thr, &c_idx, &is_flipped)
                labels[i] = c_idx

                # Update centroids data structure to account for the new streamline
                if c_idx < n_clusters:
                    # Update corresponding centroid
                    n1 = cluster_size[c_idx]
                    n2 = 1.0 / (n1 + 1.0)
                    if is_flipped==False:
                        for j in range(n_pts):
                            centroids[c_idx,j,0] = (n1 * <double>centroids[c_idx,j,0] + streamline[j,0]) * n2
                            centroids[c_idx,j,1] = (n1 * <double>centroids[c_idx,j,1] + streamline[j,1]) * n2
                            centroids[c_idx,j,2] = (n1 * <double>centroids[c_idx,j,2] + streamline[j,2]) * n2
                    else:
                        if metric != 'ASEDdct':
                            for j in range(n_pts):
                                centroids[c_idx,j,0] = (n1 * <double>centroids[c_idx,j,0] + streamline[n_pts-1-j,0]) * n2
                                centroids[c_idx,j,1] = (n1 * <double>centroids[c_idx,j,1] + streamline[n_pts-1-j,1]) * n2
                                centroids[c_idx,j,2] = (n1 * <double>centroids[c_idx,j,2] + streamline[n_pts-1-j,2]) * n2
                        else:
                            for j in range(n_dct):
                                if j%2==0:
                                    centroids[c_idx,j,0] = (n1 * <double>centroids[c_idx,j,0] + streamline[j,0]) * n2
                                    centroids[c_idx,j,1] = (n1 * <double>centroids[c_idx,j,1] + streamline[j,1]) * n2
                                    centroids[c_idx,j,2] = (n1 * <double>centroids[c_idx,j,2] + streamline[j,2]) * n2
                                else:
                                    centroids[c_idx,j,0] = (n1 * <double>centroids[c_idx,j,0] - streamline[j,0]) * n2
                                    centroids[c_idx,j,1] = (n1 * <double>centroids[c_idx,j,1] - streamline[j,1]) * n2
                                    centroids[c_idx,j,2] = (n1 * <double>centroids[c_idx,j,2] - streamline[j,2]) * n2
                    cluster_size[c_idx] += 1
                else:
                    # Add a new centroid
                    if n_clusters % chunk_size == 0:
                        # increase the memory for centroids
                        chunk_size += chunk_size
                        centroids_np = np.resize( centroids_np, (chunk_size, centroids_np.shape[1], 3) )
                        centroids = centroids_np
                        cluster_size_np = np.resize( cluster_size_np, (chunk_size) )
                        cluster_size = cluster_size_np
                    cluster_size[n_clusters] = 1
                    for j in range(streamline.shape[0]):
                        centroids[n_clusters, j,0] = streamline[j,0]
                        centroids[n_clusters, j,1] = streamline[j,1]
                        centroids[n_clusters, j,2] = streamline[j,2]
                    n_clusters += 1
                pbar.update()
        TCK_in.close()
        logger.subinfo(f"Number of clusters: {n_clusters}", indent_char='-', indent_lvl=2)

        # save clustered tractogram to file
        if ret_centroids==True:
            # save the centroid computed by the algorithm (with n_pts points)
            logger.subinfo(f"Saving centroids:", indent_char='*', indent_lvl=1, with_progress=True)
            TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
            with ProgressBar(total=n_clusters, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
                for i in range(n_clusters):
                    TCK_out.write_streamline( np.asarray(centroids[i],dtype=np.float32), n_pts )
            TCK_out.close( write_eof=True, count=n_clusters)
        else:
            # locate the closest streamline to each centroid (a.k.a. medoid) to be saved as representative of the corresponding cluster
            logger.subinfo(f"Saving medoids:", indent_char='*', indent_lvl=1, with_progress=True)
            medoid_idx = np.empty(n_clusters, dtype=np.int32) # index of the streamline that will represent the cluster
            closest_streamline_distance = 1e9 * np.ones(n_clusters, dtype=np.float64)
            TCK_in = LazyTractogram( tractogram_filename, mode='r' )
            with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
                for i in range(n_streamlines):
                    c_idx = labels[i]
                    TCK_in.read_streamline()
                    set_number_of_points_f64(TCK_in.streamline, TCK_in.n_pts, streamline_res, n_pts, lengths)
                    if metric != 'ASEDdct':
                        for j in range(n_pts):
                            streamline[j,0] = streamline_res[j,0]
                            streamline[j,1] = streamline_res[j,1]
                            streamline[j,2] = streamline_res[j,2]
                    else:
                        tmp = np.dot( dct_M, streamline_res )
                        for j in range(n_dct):
                            streamline[j,0] = tmp[j,0]
                            streamline[j,1] = tmp[j,1]
                            streamline[j,2] = tmp[j,2]

                    for j in range(centroid.shape[0]):
                        centroid[j,0] = centroids[c_idx,j,0]
                        centroid[j,1] = centroids[c_idx,j,1]
                        centroid[j,2] = centroids[c_idx,j,2]

                    d = distance.calculate(streamline, streamline_res, &is_flipped)
                    if d<closest_streamline_distance[c_idx] or fabs(d-closest_streamline_distance[c_idx]) < 1e-6: #FIXME: remove the second check
                        closest_streamline_distance[c_idx] = d
                        medoid_idx[c_idx] = i
                    pbar.update()
            TCK_in.close()

            medoid_idx_indices = np.argsort( np.asarray(medoid_idx) )
            medoid_idx_sorted = np.asarray(medoid_idx)[ medoid_idx_indices ]
            TCK_in = LazyTractogram( tractogram_filename, mode='r' )
            TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
            c_idx = 0
            for i in range(n_streamlines):
                TCK_in.read_streamline()
                if i == medoid_idx_sorted[c_idx]:
                    TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
                    c_idx += 1
                    if c_idx==n_clusters:
                        break
            TCK_out.close(write_eof=True, count=c_idx)
            TCK_in.close()
            if c_idx != n_clusters:
                logger.error( f'Written only {c_idx} streamlines to file (<{n_clusters})' )

            if out_labels_filename is not None:
                tmp = medoid_idx_indices[labels].astype(dtype=np.uint32)
                if out_labels_filename.endswith('.txt'):
                    np.savetxt(out_labels_filename, tmp, fmt='%d')
                else:
                    np.save(out_labels_filename, tmp, allow_pickle=False)

    except Exception as e:
        if os.path.isfile( out_tractogram_filename ):
            os.remove( out_tractogram_filename )
        if (out_labels_filename is not None) and os.path.isfile( out_labels_filename ):
            os.remove( out_labels_filename )
        logger.error( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        logger.info( f'[ {format_time(time() - t0)} ]' )
        return