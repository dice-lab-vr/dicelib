#ifndef STREAMLINE_UTILS_H
#define STREAMLINE_UTILS_H

#include "Catmull.h"
#include "psimpl_v7_src/psimpl.h"
#include "Vector.h"
#include <vector>
#include <iterator>


// =========================
// Function called by CYTHON
// =========================
int smooth_c( float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len )
{
    std::vector<float>          polyline_simplified;
    std::vector<Vector<float>>  CPs;
    Catmull                     FIBER;
    int                         n;

    if ( nP<=2 )
    {
        // if input streamline has less than 2 points, just copy input to output
        for( int j=0; j<3*nP; j++ )
            *(ptr_npaFiberO++) = *(ptr_npaFiberI++);
        return nP;
    }
    else
    {
        // check that at least 3 points are considered
        n = nP*ratio;
        if ( n<3 )
            n = 3;

        // simplify input polyline down to n points
        psimpl::simplify_douglas_peucker_n<3>( ptr_npaFiberI, ptr_npaFiberI+3*nP, n, std::back_inserter(polyline_simplified) );

        CPs.resize( polyline_simplified.size()/3 );
        for( int j=0,index=0; j < polyline_simplified.size(); j=j+3 )
            CPs[index++].Set( polyline_simplified[j], polyline_simplified[j+1], polyline_simplified[j+2] );

        // perform interpolation
        FIBER.set( CPs );
        FIBER.eval( FIBER.L/segment_len );
        FIBER.arcLengthReparametrization( segment_len );

        // copy coordinates of the smoothed streamline back to python
        for( int j=0; j<FIBER.P.size(); j++ )
        {
            *(ptr_npaFiberO++) = FIBER.P[j].x;
            *(ptr_npaFiberO++) = FIBER.P[j].y;
            *(ptr_npaFiberO++) = FIBER.P[j].z;
        }
        return FIBER.P.size();
    }
}


int rdp_red_c( float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float epsilon, int n_pts_red )
{
    std::vector<float>          polyline_simplified;
    int                         n_out;

    if ( nP<=2 )
    {
        // if input streamline has less than 2 points, just copy input to output
        for( int j=0; j<3*nP; j++ )
            *(ptr_npaFiberO++) = *(ptr_npaFiberI++);
        return nP;
    }
    else
    {
        if ( n_pts_red>0 )
        {
            // simplify input polyline using to n points
            psimpl::simplify_douglas_peucker_n<3>( ptr_npaFiberI, ptr_npaFiberI+3*nP, n_pts_red, std::back_inserter(polyline_simplified) );
        }
        else
        {
            // simplify input polyline 
            psimpl::simplify_douglas_peucker<3>( ptr_npaFiberI, ptr_npaFiberI+3*nP, epsilon, std::back_inserter(polyline_simplified) );
        }

        for( int j=0; j<polyline_simplified.size(); j++ )
            *(ptr_npaFiberO++) = polyline_simplified[j];
        n_out = polyline_simplified.size()/3;

        return n_out;
    }
}


int create_replicas_point( float* ptr_pts_in, double* ptr_pts_out, double* ptr_blur_rho, double* ptr_blur_angle, int n_replicas, float fiberShiftX, float fiberShiftY, float fiberShiftZ )
{
    // From trk2dictionary, few little changes

    thread_local static Vector<double> S1, S2, q, n, nr, qxn, qxqxn, dir2;
    thread_local static double         alpha, w, R;
    thread_local static int            k;
    std::vector<double>                coord_replicas;

    // create duplicate points
    S1.x = ptr_pts_in[0]+fiberShiftX;
    S1.y = ptr_pts_in[1]+fiberShiftY;
    S1.z = ptr_pts_in[2]+fiberShiftZ;
    dir2.x = (ptr_pts_in[3]+fiberShiftX) - S1.x;
    dir2.y = (ptr_pts_in[4]+fiberShiftY) - S1.y;
    dir2.z = (ptr_pts_in[5]+fiberShiftZ) - S1.z;
    dir2.Normalize();
    n.x = dir2.y-dir2.z;
    n.y = dir2.z-dir2.x;
    n.z = dir2.x-dir2.y;
    n.Normalize();

    // duplicate first point and move to corresponding grid locations
    for(k=0; k<n_replicas ;k++)
    {
        R = ptr_blur_rho[k];
        alpha = ptr_blur_angle[k];

        // quaternion (q.x, q.y, q.z, w) for rotation
        w = sin(alpha/2.0);
        q.x = dir2.x * w;
        q.y = dir2.y * w;
        q.z = dir2.z * w;
        w = cos(alpha/2.0);

        // rotate the segment's normal
        qxn.x = 2.0 * ( q.y * n.z - q.z * n.y );
        qxn.y = 2.0 * ( q.z * n.x - q.x * n.z );
        qxn.z = 2.0 * ( q.x * n.y - q.y * n.x );
        qxqxn.x = q.y * qxn.z - q.z * qxn.y;
        qxqxn.y = q.z * qxn.x - q.x * qxn.z;
        qxqxn.z = q.x * qxn.y - q.y * qxn.x;
        nr.x = n.x + w * qxn.x + qxqxn.x;
        nr.y = n.y + w * qxn.y + qxqxn.y;
        nr.z = n.z + w * qxn.z + qxqxn.z;
        nr.Normalize();

        // move first point to corresponding grid location
        *(ptr_pts_out++) = S1.x + R*nr.x;
        *(ptr_pts_out++) = S1.y + R*nr.y;
        *(ptr_pts_out++) = S1.z + R*nr.z;
    }

    return n_replicas;
}


void create_replicas_streamline( float* fiber, unsigned int pts, float* fiber_out, float* pts_replica, int nReplicas, double* ptrBlurRho, double* ptrBlurAngle, double* ptrBlurWeights, bool doApplyBlur) 
{

    float *P = new float[nReplicas*3] {0};

    // From trk2dictionary_c.cpp (function fiberForwardModel) with few changes to save the output in fiber_out

    thread_local static Vector<double> S1, S2, S1m, S2m, P_old, P_tmp, q, n, nr, qxn, qxqxn;
    thread_local static Vector<double> vox, vmin, vmax, dir1, dir2;
    thread_local static double         len, t, alpha, w, R, dot;
    thread_local static int            i, j, k, ii;

    if ( pts <= 2 )
        return;

    // create duplicate points
    S1.x = fiber[0];
    S1.y = fiber[1];
    S1.z = fiber[2];
    dir2.x = fiber[3] - S1.x;
    dir2.y = fiber[4] - S1.y;
    dir2.z = fiber[5] - S1.z;
    dir2.Normalize();
    n.x = dir2.y-dir2.z;
    n.y = dir2.z-dir2.x;
    n.z = dir2.x-dir2.y;
    n.Normalize();

    // duplicate first point and move to corresponding grid locations
    for(k=0; k<nReplicas ;k++)
    {
        if ( !doApplyBlur && k>0 )
            continue;
        R = ptrBlurRho[k];
        alpha = ptrBlurAngle[k];

        // quaternion (q.x, q.y, q.z, w) for rotation
        w = sin(alpha/2.0);
        q.x = dir2.x * w;
        q.y = dir2.y * w;
        q.z = dir2.z * w;
        w = cos(alpha/2.0);

        // rotate the segment's normal
        qxn.x = 2.0 * ( q.y * n.z - q.z * n.y );
        qxn.y = 2.0 * ( q.z * n.x - q.x * n.z );
        qxn.z = 2.0 * ( q.x * n.y - q.y * n.x );
        qxqxn.x = q.y * qxn.z - q.z * qxn.y;
        qxqxn.y = q.z * qxn.x - q.x * qxn.z;
        qxqxn.z = q.x * qxn.y - q.y * qxn.x;
        nr.x = n.x + w * qxn.x + qxqxn.x;
        nr.y = n.y + w * qxn.y + qxqxn.y;
        nr.z = n.z + w * qxn.z + qxqxn.z;
        nr.Normalize();

        // move first point to corresponding grid location
        S2.x = S1.x + R*nr.x;
        S2.y = S1.y + R*nr.y;
        S2.z = S1.z + R*nr.z;
        P[k*3+0] = S2.x;
        P[k*3+1] = S2.y;
        P[k*3+2] = S2.z;

        // save the first point of the k-th replica
        fiber_out[k*pts*3+0] = S2.x;
        fiber_out[k*pts*3+1] = S2.y;
        fiber_out[k*pts*3+2] = S2.z;
        pts_replica[k] += 1;
    }

    // move all remaining points
    for(i=1; i<pts ;i++)
    {
        /* get the intersection plane */
        // S2 = point on plane
        S2.x = fiber[i*3+0];
        S2.y = fiber[i*3+1];
        S2.z = fiber[i*3+2];

        // n = normal to plane
        dir1.x = S2.x - fiber[(i-1)*3+0];
        dir1.y = S2.y - fiber[(i-1)*3+1];
        dir1.z = S2.z - fiber[(i-1)*3+2];
        dir1.Normalize();
        if ( i == pts-1 )
        {
            dir2.x = dir1.x;
            dir2.y = dir1.y;
            dir2.z = dir1.z;
        } else {
            dir2.x = fiber[(i+1)*3+0] - S2.x;
            dir2.y = fiber[(i+1)*3+1] - S2.y;
            dir2.z = fiber[(i+1)*3+2] - S2.z;
            dir2.Normalize();
        }
        n.x = 0.5*(dir1.x+dir2.x);
        n.y = 0.5*(dir1.y+dir2.y);
        n.z = 0.5*(dir1.z+dir2.z);

        // normalize to avoid computations later on
        dot = dir1.x*n.x + dir1.y*n.y + dir1.z*n.z;
        n.x /= dot;
        n.y /= dot;
        n.z /= dot;

        /* translate points */
        for(k=0; k<nReplicas ;k++)
        {
            if ( !doApplyBlur && k>0 )
                continue;

            if ( ptrBlurWeights[k] < 1e-3 )
                continue;

            P_old.x = P[k*3+0];
            P_old.y = P[k*3+1];
            P_old.z = P[k*3+2];
            len = (S2.x-P_old.x)*n.x + (S2.y-P_old.y)*n.y + (S2.z-P_old.z)*n.z;
            if ( len>0 )
            {
                P[k*3+0] += dir1.x * len;
                P[k*3+1] += dir1.y * len;
                P[k*3+2] += dir1.z * len;

                // save the i-th point of k-th replica
                ii = pts_replica[k];
                fiber_out[k*pts*3+ii*3+0] = P[k*3+0];
                fiber_out[k*pts*3+ii*3+1] = P[k*3+1];
                fiber_out[k*pts*3+ii*3+2] = P[k*3+2];
                pts_replica[k] += 1;

            }
        }
    }
    delete[] P;
}


#endif
