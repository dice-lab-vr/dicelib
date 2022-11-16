#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython
import numpy as np
cimport numpy as np
cimport cython 
from libc.math cimport sqrt , round , floor 


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


cpdef sampling(double [:,::1] streamline_view, img, int npoints,option = None):
    """Compute the length of a streamline.

    Parameters
    ----------
    streamline : Nx3 numpy array
        The streamline data, coordinates 
    img : numpy array 
        data of the image 
    npoints : int  
        points of the streamline 
    Returns
    -------
    value : numpy array of dim (npoint,)
        values that correspond to coordinates of streamline in the image space 
         
    """
    if not img.flags['C_CONTIGUOUS']: # check img is c contiguous 
        img = np.ascontiguousarray(img)
    
    cdef double [:,:,::1] img_view = img #def cython memoryview , c_contiguous 
    value = np.empty([npoints,], dtype= float) 
    opt_value = np.empty([1,], dtype= float)
    cdef size_t ii 
    for ii in range(npoints):      
        value[ii] = img_view[<int>round(streamline_view[ii,0]),<int>round(streamline_view[ii,1]),<int>round(streamline_view[ii,2])] #cast int values 
         
        
    if option == "mean":
        opt_value[0] = value.sum()/npoints
        return opt_value
    elif option == "median":
        opt_value[0] = np.median(value)
        return opt_value
    elif option == "min":
        opt_value[0] = value.min()
        return opt_value
    elif option == "max":
        opt_value[0] = value.max()
        return opt_value
    else: #none case
        return value 
