#!python
# cython: boundscheck=False, wraparound=False, profile=False
import cython 
import numpy as np
from . import ui
cimport numpy as np
cimport cython
from libc.math cimport floor 

cpdef double [:,::1] space_tovox(streamline,header,curr_space = None ):
    """Method to change space reference of streamlines (.tck)
    Note that if curr_space is None, space is interpreted as RASmm 

    Allowed spaces tranformation: 
    voxmm --> vox 
    rasmm --> vox

    Parameters:

    -----------

    streamline : Numpy array Nx3
        Data of the streamline, coordinates 
    header : NiftiHeader 
        header of the image
    curr space : string 
        coordinates space of streamline to transform
    """
    streamline = np.asarray(streamline)
    voxsize = np.asarray(header["pixdim"][1:4]) #resolution
    affine = np.asarray([header["srow_x"],header["srow_y"],header["srow_z"],[0,0,0,1]]) #affine retrieved from the header
    inverse = np.linalg.inv(affine) #inverse of affine
    small = inverse[:-1,:-1].T 
    val = inverse[:-1,-1]
    # if curr_space == "voxmm":
    #     streamline /= voxsize
    # elif curr_space == "rasmm": 
    #     streamline = np.matmul(streamline,inverse[:-1,:-1].T) + inverse[:-1,-1] #same as nibabel.affines.apply_affine() 
    #     streamline += voxsize/2 #to point center of the voxel 
    #     streamline = np.floor(streamline) #cast 


    if not streamline.flags['C_CONTIGUOUS']:
        streamline = np.ascontiguousarray(streamline)
    if not small.flags['C_CONTIGUOUS']:
        small = np.ascontiguousarray(small)
    if not val.flags['C_CONTIGUOUS']:
        val = np.ascontiguousarray(val)
    if not small.flags['C_CONTIGUOUS']:
        small = np.ascontiguousarray(small)

    cdef double [:,::1] streamline_view = np.double(streamline)
    cdef double [:,::1] small_view = small
    cdef float  [::1] voxsize_view = voxsize
    cdef double [::1] val_view = val  
    cdef double somma = 0.0
    cdef size_t ii, yy

    if curr_space == "voxmm":
        for ii in range(streamline_view.shape[0]):
            for yy in range(streamline_view.shape[1]):
                streamline_view[ii][yy] = streamline_view[ii][yy]/voxsize_view[yy]
    else :     #rasmm 
        for ii in range(streamline_view.shape[0]):
            for yy in range(streamline_view.shape[1]):
                somma = ((streamline_view[ii,0]*small_view[0,yy] + streamline_view[ii,1]*small_view[1,yy] + streamline_view[ii,2]*small_view[2,yy]) + val_view[yy]) 
                somma += (voxsize_view[yy]/2)
                streamline_view[ii][yy] = floor(somma)
    
    return streamline_view