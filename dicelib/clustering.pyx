# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, language_level=3
from dicelib.streamline cimport set_number_of_points
from dicelib.ui import ProgressBar, set_verbose, setup_logger
from dicelib.utils import check_params, Dir, File, Num, format_time
import os
from time import time
import numpy as np
cimport numpy as np
from dicelib.tractogram cimport LazyTractogram
from libc.math cimport sqrtf

logger = setup_logger('clustering')


cdef class DistanceMetric:
    """Base class to define new distance metrics; new metrics should inherit from this class.
    All the methods need to be overloaded to account for the specific needs of the metric.
    Each method will then be called by a dispatcher when needed.

    Attributes
    ----------
    n_pts : integer
        The number of coordinates each streamline should have to compute the given distance.
    """
    cdef:
        int n_pts

    def __init__( self, int n_pts ) :
        self.n_pts = n_pts
        return

    cdef float calculate(self, float[:,::1] f, float[:,::1] g, int* out_flipped) nogil:
        raise NotImplementedError("Subclasses must implement this function")

    cdef void closest_centroid(self, float[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_belongs_to, int* out_flipped) nogil:
        raise NotImplementedError("Subclasses must implement this function")

    def __dealloc__(self):
        pass


cdef class AverageSquaredEuclideanDistance(DistanceMetric):
    """Average Squared Euclidean Distance (ASED) between streamlines."""

    cdef float calculate(self, float[:,::1] f, float[:,::1] g, int* out_flipped) nogil:
        """Calculate the distance between two streamlines"""
        cdef:
            size_t j, k
            float dx, dy, dz
            float dist_direct = 0, dist_flipped = 0

        for j in range(self.n_pts):
            dx = f[j, 0] - g[j, 0]
            dy = f[j, 1] - g[j, 1]
            dz = f[j, 2] - g[j, 2]
            dist_direct += dx*dx + dy*dy + dz*dz

            k = self.n_pts-j-1
            dx = f[k, 0] - g[j, 0]
            dy = f[k, 1] - g[j, 1]
            dz = f[k, 2] - g[j, 2]
            dist_flipped += dx*dx + dy*dy + dz*dz

        if dist_direct <= dist_flipped:
            out_flipped[0] = 0
            return dist_direct
        else:
            out_flipped[0] = 1
            return dist_flipped


    cdef void closest_centroid(self, float[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_belongs_to, int* out_flipped) nogil:
        """Calculate the distance between a streamline and a set of centroids"""
        cdef:
            size_t i, j, k
            float dx, dy, dz
            float dist_direct, dist_flipped
            float dist_min_all = self.n_pts * thr
            int belongs_to = n_clusters, flipped

        for i in range(n_clusters):
            dist_direct = 0
            dist_flipped = 0
            for j in range(self.n_pts):
                dx = streamline[j, 0] - centroids[i, j, 0]
                dy = streamline[j, 1] - centroids[i, j, 1]
                dz = streamline[j, 2] - centroids[i, j, 2]
                dist_direct += dx*dx + dy*dy + dz*dz

                k = self.n_pts-j-1
                dx = streamline[k, 0] - centroids[i, j, 0]
                dy = streamline[k, 1] - centroids[i, j, 1]
                dz = streamline[k, 2] - centroids[i, j, 2]
                dist_flipped += dx*dx + dy*dy + dz*dz

                # if both direct and flipped distances are already worse
                # than best found distance, no need to continue computing
                if j % 4 == 0:
                    if dist_direct >= dist_min_all and dist_flipped >= dist_min_all:
                        break

            # only update if a new minimum is found
            if dist_direct <= dist_flipped:
                if dist_direct < dist_min_all:
                    dist_min_all = dist_direct
                    flipped = 0
                    belongs_to = i
            else:
                if dist_flipped < dist_min_all:
                    dist_min_all = dist_flipped
                    flipped = 1
                    belongs_to = i

        out_belongs_to[0] = belongs_to
        out_flipped[0] = flipped
        return


cdef class AverageEuclideanDistance(DistanceMetric):
    """Average Euclidean Distance (AED) between streamlines"""

    cdef float calculate(self, float[:,::1] f, float[:,::1] g, int* out_flipped) nogil:
        """Calculate the distance between two streamlines"""
        cdef:
            size_t j, k
            float dx, dy, dz
            float dist_direct = 0, dist_flipped = 0

        for j in range(self.n_pts):
            dx = f[j, 0] - g[j, 0]
            dy = f[j, 1] - g[j, 1]
            dz = f[j, 2] - g[j, 2]
            dist_direct += sqrtf( dx*dx + dy*dy + dz*dz )

            k = self.n_pts-j-1
            dx = f[k, 0] - g[j, 0]
            dy = f[k, 1] - g[j, 1]
            dz = f[k, 2] - g[j, 2]
            dist_flipped += sqrtf( dx*dx + dy*dy + dz*dz )

        # Only update if we actually found a new minimum
        if dist_direct <= dist_flipped:
            out_flipped[0] = 0
            return dist_direct
        else:
            out_flipped[0] = 1
            return dist_flipped


    cdef void closest_centroid(self, float[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_belongs_to, int* out_flipped) nogil:
        """Calculate the distance between a streamline and a set of centroids"""
        cdef:
            size_t i, j, k
            float dx, dy, dz
            float dist_direct, dist_flipped
            float dist_min_all = self.n_pts * thr
            int belongs_to = n_clusters, flipped

        for i in range(n_clusters):
            dist_direct = 0
            dist_flipped = 0
            for j in range(self.n_pts):
                dx = streamline[j, 0] - centroids[i, j, 0]
                dy = streamline[j, 1] - centroids[i, j, 1]
                dz = streamline[j, 2] - centroids[i, j, 2]
                dist_direct += sqrtf( dx*dx + dy*dy + dz*dz )

                k = self.n_pts-j-1
                dx = streamline[k, 0] - centroids[i, j, 0]
                dy = streamline[k, 1] - centroids[i, j, 1]
                dz = streamline[k, 2] - centroids[i, j, 2]
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
                    belongs_to = i
            else:
                if dist_flipped < dist_min_all:
                    dist_min_all = dist_flipped
                    flipped = 1
                    belongs_to = i

        out_belongs_to[0] = belongs_to
        out_flipped[0] = flipped
        return


cpdef cluster( str tractogram_filename, float thr, str out_tractogram_filename,
               str metric="ASED", int n_points=12, bool ret_centroids=False,
               str out_labels_filename=None, int chunk_size=10000,
               bool force=False, int verbose=3 ):
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
    metric : {'AED','ASED'}, default='ASED'
        Metric to use for computing distances between streamlines:
        - AED = Average Euclidean Distance
        - ASED = Average Squared Euclidean Distance
    n_points : int, default=12
        Number of points to resample the streamlines before clustering.
        NB: this clustering algorithm requires all streamlines to have the same number of points.
    ret_centroids : boolean, default=False
        Whether to return the centroids (i.e. mean streamline is a cluster) or medoids (i.e. closest streamline
        to a centroid) as cluster representatives.
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
        int n_streamlines, n_clusters, n_pts = n_points
        float [:,:,::1] centroids
        int [::1] medoid_idx
        int [:] cluster_size
        int centroid_chunk_size = chunk_size
        int centroids_in_mem = centroid_chunk_size
        int [:] centroid_n_pts
        float[:,::1] streamline = np.empty((n_pts,3), dtype=np.float32)
        int [:] belongs_to
        int c_idx, is_flipped
        size_t i, j, k
        float n1, n2, d
        float [:] closest_streamline_distance
        DistanceMetric distance
        LazyTractogram TCK_in = None, TCK_out = None
        float[:] lengths = np.empty(3000, dtype=np.float32)

    t0 = time()
    set_verbose('clustering', verbose)
    logger.info('Clustering tractogram')

    if not os.path.isfile(tractogram_filename):
        logger.error(f"File '{tractogram_filename}' not found")
        return
    if metric not in ['AED', 'ASED']:
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
        Num(name='n_pts', value=n_pts, min_=2),
        Num(name='chunk_size', value=chunk_size, min_=100),
    ]
    check_params(files=files, nums=nums, force=force)

    try:
        if metric == 'AED':
            distance = AverageEuclideanDistance( n_pts )
        elif metric == 'ASED':
            distance = AverageSquaredEuclideanDistance( n_pts )

        TCK_in = LazyTractogram( tractogram_filename, mode='r' )
        n_streamlines = int(TCK_in.header['count'])
        logger.subinfo(f'Input:', indent_lvl=1, indent_char='*')
        logger.subinfo(f'Number of streamlines: {n_streamlines}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Points per streamline: {n_pts}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Distance metric: {metric}', indent_lvl=2, indent_char='-')
        logger.subinfo(f'Distance threshold: {thr}', indent_lvl=2, indent_char='-')
        logger.debug(f'Memory for centroids is dynamically incremented in chunks of {chunk_size} items')
        if n_streamlines == 0:
            return

        logger.subinfo(f"Clustering:", indent_char='*', indent_lvl=1, with_progress=True)

        # initialize data structures
        centroids_np = np.empty((centroids_in_mem, n_pts, 3), dtype=np.float32)
        centroids = centroids_np
        cluster_size_np = np.ones(centroids_in_mem, dtype=np.int32)
        cluster_size = cluster_size_np
        belongs_to = np.empty(n_streamlines, dtype=np.int32)

        # Process first streamline
        n_clusters = 1
        belongs_to[0] = 0
        TCK_in.read_streamline()
        set_number_of_points(TCK_in.streamline, TCK_in.n_pts, centroids[0], n_pts, lengths)

        # Process remaining streamlines
        with ProgressBar(total=n_streamlines-1, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
            for i in range(1, n_streamlines):
                TCK_in.read_streamline()
                set_number_of_points(TCK_in.streamline, TCK_in.n_pts, streamline, n_pts, lengths)

                # Find the closest centroid
                distance.closest_centroid(streamline, centroids, n_clusters, thr, &c_idx, &is_flipped)
                belongs_to[i] = c_idx

                # Update centroids data structure to account for the new streamline
                if c_idx < n_clusters:
                    # Update corresponding centroid
                    n1 = cluster_size[c_idx]
                    n2 = 1.0 / (n1 + 1.0)
                    if is_flipped:
                        for j in range(n_pts):
                            centroids[c_idx, j, 0] = (n1 * centroids[c_idx, j, 0] + streamline[n_pts-1-j, 0]) * n2
                            centroids[c_idx, j, 1] = (n1 * centroids[c_idx, j, 1] + streamline[n_pts-1-j, 1]) * n2
                            centroids[c_idx, j, 2] = (n1 * centroids[c_idx, j, 2] + streamline[n_pts-1-j, 2]) * n2
                    else:
                        for j in range(n_pts):
                            centroids[c_idx, j, 0] = (n1 * centroids[c_idx, j, 0] + streamline[j, 0]) * n2
                            centroids[c_idx, j, 1] = (n1 * centroids[c_idx, j, 1] + streamline[j, 1]) * n2
                            centroids[c_idx, j, 2] = (n1 * centroids[c_idx, j, 2] + streamline[j, 2]) * n2
                    cluster_size[c_idx] += 1
                else:
                    # Add a new centroid
                    if n_clusters % centroid_chunk_size == 0:
                        # inrcease the memory for centroids
                        centroids_in_mem += centroid_chunk_size
                        centroids_np = np.resize( centroids_np, (centroids_in_mem, n_pts, 3) )
                        centroids = centroids_np
                        cluster_size_np = np.resize( cluster_size_np, (centroids_in_mem) )
                        cluster_size = cluster_size_np
                    cluster_size[n_clusters] = 1
                    centroids[n_clusters, :, :] = streamline[:]
                    n_clusters += 1
                pbar.update()
        TCK_in.close()
        logger.subinfo(f"Number of clusters: {n_clusters}", indent_char='-', indent_lvl=2)

        # locate the closest streamline to each centroid (a.k.a. medoid) to be saved as representative of the corresponding cluster

        # save clustered tractogram to file
        if ret_centroids==True:
            logger.subinfo(f"Saving centroids:", indent_char='*', indent_lvl=1, with_progress=True)
            TCK_out = LazyTractogram(out_tractogram_filename, mode='w', header=TCK_in.header)
            with ProgressBar(total=n_clusters, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
                for i in range(n_clusters):
                    TCK_out.write_streamline( centroids[i], n_pts )
            TCK_out.close( write_eof=True, count=n_clusters)

        else:
            logger.subinfo(f"Saving medoids:", indent_char='*', indent_lvl=1, with_progress=True)
            medoid_idx = np.empty(n_clusters, dtype=np.int32) # index of the streamline that will represent the cluster
            closest_streamline_distance = 1e9 * np.ones(n_clusters, dtype=np.float32)
            TCK_in = LazyTractogram( tractogram_filename, mode='r' )
            with ProgressBar(total=n_streamlines, disable=verbose<3, hide_on_exit=False, subinfo=True) as pbar:
                for i in range(n_streamlines):
                    TCK_in.read_streamline()
                    set_number_of_points(TCK_in.streamline, TCK_in.n_pts, streamline, n_pts, lengths)
                    c_idx = belongs_to[i]

                    d = distance.calculate(streamline, centroids[c_idx, :, :], &is_flipped)
                    if d < closest_streamline_distance[c_idx]:
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
                tmp = medoid_idx_indices[belongs_to].astype(dtype=np.uint32)
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