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


int create_replicas_pt( float* ptr_pts_in, double* ptr_pts_out, double* ptr_blur_rho, double* ptr_blur_angle, int n_replicas, float fiberShiftX, float fiberShiftY, float fiberShiftZ )
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

#endif
