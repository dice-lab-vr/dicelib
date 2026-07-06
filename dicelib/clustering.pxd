# cython: boundscheck=False, wraparound=False, profile=False, language_level=3
cdef class DistanceMetric:
    cdef:
        int n_pts
    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil
    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_labels, int* out_flipped) nogil

cdef class AverageEuclideanDistance(DistanceMetric):
    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil
    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_labels, int* out_flipped) nogil

cdef class AverageSquaredEuclideanDistance(DistanceMetric):
    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil
    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_labels, int* out_flipped) nogil

cdef class AverageSquaredEuclideanDistanceDCT(DistanceMetric):
    cdef:
        int n_dct
    cdef double calculate(self, double[:,::1] f, double[:,::1] g, int* out_flipped) nogil
    cdef double closest_centroid(self, double[:,::1] streamline, float[:,:,::1] centroids, int n_clusters, float thr, int* out_labels, int* out_flipped) nogil