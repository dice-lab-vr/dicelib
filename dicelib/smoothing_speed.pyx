from bisect import bisect_right

import numpy as _np
cimport numpy as np


cdef void compute_tangent(float[:] pt1, float[:] grid, float[:] tangent):
    cdef float x0, x1, x2 = pt1[0], pt1[1], pt1[2]
    cdef float t0, t1, t2 = grid[0], grid[1], grid[2]
    cdef float delta0 = t2 -t1
    cdef float delta1 = t1 - t0
    cdef float v0 = (x2 - x1) / delta0
    cdef float v1 = (x1 - x0) / delta1
    tangent = (delta0 * v1 + delta1 * v0) / (delta0 + delta1)


cdef float[:,:,:] CubicHermite(vertices, alpha=0.5):

    cdef size_t i = 0
    cdef float alpha = float(alpha)
    cdef float[:,:] vertices
    cdef int[:] grid
    cdef float[:,:] tangent = np.empty((vertices.shape[0] + 2 , 3), dtype=float)
    cdef float[:,:] tangents = np.empty((2*tangent.shape[0], 3), dtype=float)
    cdef float[:,:] matrix = np.array([ [2, -2, 1, 1],
                                        [-3, 3, -2, -1],
                                        [0, 0, 1, 0],
                                        [1, 0, 0, 0]]).astype(float)

    cdef float[:,:,::1] vertices = np.asarray(vertices, dtype=float)
    cdef float[:,:,::1] segments = np.empty((vertices.shape[0]-1, 4, 3), dtype=float)
    cdef int [:,:] grid = np.empty(vertices.shape[0])
    cdef float [:,:] smoothed = np.empty((vertices.shape[0]-1, 4, 3), dtype=float)


    vertices = np.asarray(vertices, dtype=float)
    grid = check_grid(grid, alpha, vertices)
    
    for i in range(vertices.shape[0]):
        compute_tangent(vertices[i], grid[i], tangent[i])


    # Calculate tangent for "natural" end condition
    x0, x1 = vertices[0], vertices[1]
    t0, t1 = grid[0], grid[1]
    delta = t1 - t0
    tangent[0] = 3 * (x1 - x0) / (2*delta) - tangent[1] / 2

    x0, x1 = vertices[tangent.shape[0]-2], vertices[tangent.shape[0]-1]
    t0, t1 = grid[grid.shape[0]-2], grid[grid.shape[0]-1]
    delta = t1 - t0
    tangent[tangent.shape[0]-1] = 3 * (x1 - x0) / (2*delta) - tangent[tangent.shape[0]-2] / 2

    for i in range(tangent.shape[0]-1):
        tangents[2*i] = tangent[i]
        tangents[2*i+1] = tangent[i+1]

    for i in range(vertices.shape[0]-1):
        x0, x1 = vertices[i], vertices[i+1]
        v0, v1 = tangents[i], tangents[i+1] # CHECK THIS!!!
        t0, t1 = grid[i], grid[i+1]
        segments[i] = np.dot(matrix, np.array([x0, x1, (t0-t1)*v0, (t1-t0)*v1]))

    for i in range(idx.shape[0]):
        if idx[i] < grid[-1]:
            idx_temp = bisect_right(grid, param) - 1
        else:
            idx_temp = len(grid) - 2

        t0, t1 = grid[idx_temp:idx_temp+2]
        t = (idx[i] - t0) / (t1 - t0)
        coeff = segments[idx_temp]
        powers = np.arange(len(coeff))[::-1]
        weights = powers + 1 
        smoothed[i] = np.dot(coeff, t**powers * weights)

    return smoothed
    
    

cdef float [:] check_grid(int[:] grid, float alpha, float[:,:] vertices):
    cdef size_t i =1
    cdef size_t ii = 0
    cdef size_t jj = 0
    if alpha == 0:
        # NB: This is the same as alpha=0, except the type is int
        for jj in range(vertices.shape[0]):
            grid[jj] = jj

    vertices = _np.asarray(vertices)
    grid[0] = [0]
    for ii in range(vertices.shape[0]-1):
        x0 = vertices[ii]
        x1 = vertices[ii+1]
        delta = _np.linalg.norm(x1 - x0)**alpha
        if delta == 0:
            raise ValueError(
                'Repeated vertices are not possible with alpha != 0')
        grid[i] = grid[-1] + delta
        i += 1
    return grid


