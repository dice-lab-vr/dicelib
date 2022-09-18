#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt


cpdef length( float [:,:] streamline, int n=0 ):
    """Compute the length of a streamline.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data
    n : int
        Writes first n points of the streamline. If n<=0 (default), writes all points

    Returns
    -------
    length : double
        Length of the streamline in mm
    """
    if n<0:
        n = streamline.shape[0]
    cdef float* ptr     = &streamline[0,0]
    cdef float* ptr_end = ptr+n*3-3
    cdef double length = 0.0
    while ptr<ptr_end:
        length += sqrt( (ptr[3]-ptr[0])**2 + (ptr[4]-ptr[1])**2 + (ptr[5]-ptr[2])**2 )
        ptr += 3
    return length




def sampling(float [:,:] streamline, float [:,:,:] img, int n = 0):
     
  
    return None 

    
#if n<0:
#        n = streamline.shape[0]
#    cdef float* ptr     = &streamline[0,0]
 
    #cdef float* ptr_end = ptr+n*3-3
    #cdef double value = 0.0
    #while ptr<ptr_end:
        #TOcomplete value = np.append(value,img[ptr[0]/res)
        #value = np.append(value,"nan")
        #ptr += 3
    #eturn value
