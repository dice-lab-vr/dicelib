#include "../include/Catmull.h"
#include "../include/psimpl_v7_src/psimpl.h"
#include "../include/Vector.h"
#include <vector>


// =========================
// Function called by CYTHON
// =========================
int do_spline_smoothing( float* ptr_npaFiberI, int nP, float* ptr_npaFiberO, float ratio, float segment_len )
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
