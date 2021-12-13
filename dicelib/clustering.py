"""Functions to perform clustering of tractograms"""

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature
from dipy.tracking.streamline import set_number_of_points


def get_streamlines_close_to_centroids( clusters, streamlines, n_pts ):
    """Return the streamlines closer to the centroids of each cluster.

    As first step, the streamlines of the input tractogram are resampled to n_pts points.
    """
    sample_streamlines = set_number_of_points(streamlines, n_pts)

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


def cluster( filename_in, filename_reference, filename_out, thresholds, n_pts=12, replace_centroids=False, random=True, verbose=False ) :
    """ Cluster streamlines in a tractogram.
    """
    if verbose :
        print( f'-> Clustering "{filename_in}":' )

    tractogram = load_tractogram( filename_in, reference=filename_reference, bbox_valid_check=False )
    if verbose :
        print( f'- {len(tractogram.streamlines)} streamlines found' )

    if np.isscalar( thresholds ) :
        thresholds = [ thresholds ]

    metric   = AveragePointwiseEuclideanMetric( ResampleFeature( nb_points=n_pts ) )

    if verbose :
        print( '- Running QuickBundlesX...' )
    if random == False :
        clusters = QuickBundlesX( thresholds, metric ).cluster( tractogram.streamlines )
    else:
        rng = np.random.RandomState()
        ordering = np.arange(len(tractogram.streamlines))
        rng.shuffle(ordering)
        clusters = QuickBundlesX( thresholds, metric ).cluster( tractogram.streamlines, ordering=ordering )
    if verbose :
        print( f'  * {len(clusters.leaves)} clusters in lowest level'  )

    if replace_centroids :
        if verbose :
            print( '- Replace centroids with closest streamline in input tractogram' )
        centroids = get_streamlines_close_to_centroids( clusters.leaves, tractogram.streamlines, n_pts )
    else :
        if verbose :
            print( '- Keeping original centroids' )
        centroids = [ leave.centroid for leave in clusters.leaves ]

    if verbose :
        print( f'- Save to "{filename_out}"' )
    tractogram_new = StatefulTractogram.from_sft( centroids, tractogram )
    save_tractogram( tractogram_new, filename_out, bbox_valid_check=False )
