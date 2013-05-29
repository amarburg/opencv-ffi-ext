
#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/core/core_c.h>

// n.b. I've changed the API.  It now assumes _m1 is an  N x 2 1-channel matrix, 
// not a N x 1 2-channel matrix.  As such, count = _m1->rows, 
// not _m1->rows * _m1->cols as before
//
// _m1 and _m2 must be the .x and .y values in normalized homogeneous points (x,y,1)
//
CV_IMPL void cvFMaxReprojError( const CvMat* _m1, const CvMat* _m2, const CvMat* model, CvMat* _err )
{
    int i, count = _m1->rows;
    const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
    const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
    const double* F = model->data.db;
    float* err = _err->data.fl;
    
    for( i = 0; i < count; i++ )
    {
        double a, b, c, d1, d2, s1, s2;

        a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
        b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
        c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

        s2 = 1./(a*a + b*b);
        d2 = m2[i].x*a + m2[i].y*b + c;

        a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
        b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
        c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

        s1 = 1./(a*a + b*b);
        d1 = m1[i].x*a + m1[i].y*b + c;

        err[i] = (float)std::max(d1*d1*s1, d2*d2*s2);
    }
}

CV_IMPL void cvHMaxReprojError( const CvMat* _m1, const CvMat* _m2, const CvMat* model, CvMat* _err )
{
    int i, count = _m1->rows;
    const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
    const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
    const double* H = model->data.db;
    float* err = _err->data.fl;
    
    for( i = 0; i < count; i++ )
    {
        double a, b, c, d1, d2, s1, s2;

        // This is H m1 
        a = H[0]*m1[i].x + H[1]*m1[i].y + H[2];
        b = H[3]*m1[i].x + H[4]*m1[i].y + H[5];
        c = H[6]*m1[i].x + H[7]*m1[i].y + H[8];

        a /= c;
        b /= c;
        d2 = (m2[i].x - a)*(m2[i].x - a) + (m2[i].y - b)*(m2[i].y - b);

        // This is H^T m2
        a = H[0]*m2[i].x + H[3]*m2[i].y + H[6];
        b = H[1]*m2[i].x + H[4]*m2[i].y + H[7];
        c = H[2]*m2[i].x + H[5]*m2[i].y + H[8];

        a /= c;
        b /= c;
        d2 = (m1[i].x - a)*(m1[i].x - a) + (m1[i].y - b)*(m1[i].y - b);

        err[i] = (float)std::max(d1, d2);
    }
}



