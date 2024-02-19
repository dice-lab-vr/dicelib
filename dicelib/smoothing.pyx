#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

from bisect import bisect_right

import numpy as np
cimport numpy as np


cdef float[:] compute_tangent(float[:,:] points, float[:] grid):
    cdef float[:] x0 = points[0]
    cdef float[:] x1 = points[1]
    cdef float[:] x2 = points[2]   
    cdef float t0 = grid[0]
    cdef float t1 = grid[1]
    cdef float t2 = grid[2]
    cdef float delta0 = t2 -t1
    cdef float delta1 = t1 - t0
    cdef float[:] v0 = np.empty((3,), dtype=np.float32)
    cdef float[:] v1 = np.empty((3,), dtype=np.float32)
    cdef float[:] tangent = np.empty((3,), dtype=np.float32)
    cdef size_t i = 0

    for i in range(3):
        v0[i] = (x2[i] - x1[i]) / delta0
        v1[i] = (x1[i] - x0[i]) / delta1
    for i in range(3):
        tangent[i] = (delta0 * v1[i] + delta1 * v0[i]) / (delta0 + delta1)
        
    return tangent


cdef float[:, :] CatmullRom_smooth(float[:, :] vertices, float[:, :] matrix, float alpha=0.5, int num_pts=10):
    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t k = 0
    cdef int idx_temp = 0
    # cdef float[:] x0 = np.empty((3,), dtype=np.float32)
    # cdef float[:] x1 = np.empty((3,), dtype=np.float32)
    # cdef float[:] v0 = np.empty((3,), dtype=np.float32)
    # cdef float[:] v1 = np.empty((3,), dtype=np.float32)
    cdef float t0 = 0
    cdef float t1 = 0

    cdef float[:] t = np.empty((num_pts,), dtype=np.float32)
    cdef float[:, :] tangent = np.empty((vertices.shape[0]-2, 3), dtype=np.float32)
    cdef float[:,:] tangents = np.empty((2*tangent.shape[0]+2, 3), dtype=np.float32)
    cdef float[:, :, :] segments = np.empty((vertices.shape[0]-1, 4, 3), dtype=np.float32)
    cdef float[:] grid = np.empty(vertices.shape[0], dtype=np.float32)
    cdef float[:,:] prod = np.empty((4, 3), dtype=np.float32)

    grid = check_grid(grid, alpha, vertices)
    cdef float[:, :] smoothed = np.empty((num_pts, 3), dtype=np.float32)

    for i in range(vertices.shape[0]):
        # compute tangent over triplets of vertices and grid points
        if i < vertices.shape[0]-2:
            tangent[i] = compute_tangent(vertices[i:i+3], grid[i:i+3])

    # fill tangents array by duplicating each value of tangent starting from the second
    for i in range(tangent.shape[0]):
        tangents[2*i+1] = tangent[i]
        tangents[2*i+2] = tangent[i]
    

    # Calculate tangent for "natural" end condition
    x0, x1 = vertices[0], vertices[1]
    t0, t1 = grid[0], grid[1]
    delta = t1 - t0

    for i in range(3):
        tangents[0][i] = 3 * (x1[i] - x0[i]) / (2*delta) - tangent[0][i] / 2

    x0, x1 = vertices[vertices.shape[0]-2], vertices[vertices.shape[0]-1]
    t0, t1 = grid[grid.shape[0]-2], grid[grid.shape[0]-1]
    delta = t1 - t0
    for i in range(3):
        tangents[tangents.shape[0]-1][i] = 3 * (x1[i] - x0[i]) / (2*delta) - tangent[tangent.shape[0]-1][i] / 2

    for i in range(vertices.shape[0]-1):
        x0 = np.asarray(vertices[i])
        x1 = np.asarray(vertices[i+1])
        v0 = np.asarray(tangents[2*i])
        v1 = np.asarray(tangents[2*i+1])
        t0 = grid[i]
        t1 = grid[i+1]
        
        # for j in range(4):
        #     for k in range(3):
        #         prod[j][k] = matrix[j][0] * x0[k] + matrix[j][1] * x1[k] + matrix[j][2] * v0[k] + matrix[j][3] * v1[k]
        prod = matrix @ np.array([x0, x1, (t1 - t0) * v0, (t1 - t0) * v1])      
        for j in range(4):
            for k in range(3):
                segments[i][j][k] = prod[j][k]

    t = np.linspace(0, np.array(grid).max(), num_pts).astype(np.float32)
    for i in range(t.shape[0]):
        if t[i] < grid[grid.shape[0]-1]:
            idx_temp = bisect_right(grid, t[i]) - 1
        else:
            idx_temp = len(grid) - 2

        t0, t1 = grid[idx_temp:idx_temp+2]
        tt = (t[i] - t0) / (t1 - t0)
        coefficients = segments[idx_temp]
        powers = np.arange(len(coefficients))[::-1]
        new_val = tt**powers @ coefficients
        for j in range(3):
            smoothed[i][j] = new_val[j]
    return smoothed


cdef float[:] check_grid(float[:] grid, float alpha, float[:, :] vertices):
    cdef size_t i = 1
    cdef size_t ii = 0
    cdef size_t jj = 0
    cdef float diff = 0
    cdef float[:] x0 = np.empty((3,), dtype=np.float32)
    cdef float[:] x1 = np.empty((3,), dtype=np.float32)

    if alpha == 0:
        # NB: This is the same as alpha=0, except the type is int
        for jj in range(vertices.shape[0]):
            grid[jj] = jj

    grid[0] = 0
    for ii in range(vertices.shape[0]-1):
        for jj in range(3):
            x0[jj] = vertices[ii][jj]
            x1[jj] = vertices[ii+1][jj]
        
        # rewrite diff to avoid numpy overhead
        diff = np.sqrt((x1[0] - x0[0])**2 + (x1[1] - x0[1])**2 + (x1[2] - x0[2])**2)**alpha
        # x0 = np.asarray(vertices[ii])
        # x1 = np.asarray(vertices[ii+1])
        # diff = np.linalg.norm(x1 - x0)**alpha

        if diff == 0:
            raise ValueError(
                'Repeated vertices are not possible with alpha != 0')
        grid[i] = grid[i-1] + diff
        i += 1
    return grid


cpdef spline_smooth(vertices, alpha=0.5, num_pts=10):
    cdef float[:, :] matrix = np.array([ [2, -2, 1, 1],
                                          [-3, 3, -2, -1],
                                          [0, 0, 1, 0],
                                          [1, 0, 0, 0]]).astype(np.float32)
    vertices = np.asarray(vertices).astype(np.float32)
    smoothed = np.asarray(CatmullRom_smooth(vertices, matrix, alpha, num_pts))

    return smoothed