/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "cvffi_modelest.h"
#include "cvffi_fundam.h"

using namespace cv;

template<typename T> int icvCompressPoints( T* ptr, const uchar* mask, int mstep, int count )
{
  int i, j;
  for( i = j = 0; i < count; i++ )
    if( mask[i*mstep] )
    {
      if( i > j )
        ptr[j] = ptr[i];
      j++;
    }
  return j;
}


HomographyEstimator::HomographyEstimator(int _modelPoints, int _max_iters )
    : CvffiModelEstimator2(_modelPoints, cvSize(3,3), 1, _max_iters)
{
    assert( _modelPoints == 4 || _modelPoints == 5 );
    checkPartialSubsets = false;
}

int HomographyEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* H )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

    double LtL[9][9], W[9][1], V[9][9];
    CvMat _LtL = cvMat( 9, 9, CV_64F, LtL );
    CvMat matW = cvMat( 9, 1, CV_64F, W );
    CvMat matV = cvMat( 9, 9, CV_64F, V );
    CvMat _H0 = cvMat( 3, 3, CV_64F, V[8] );
    CvMat _Htemp = cvMat( 3, 3, CV_64F, V[7] );
    CvPoint2D64f cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count; cm.y /= count;
    cM.x /= count; cM.y /= count;

    for( i = 0; i < count; i++ )
    {
        sm.x += fabs(m[i].x - cm.x);
        sm.y += fabs(m[i].y - cm.y);
        sM.x += fabs(M[i].x - cM.x);
        sM.y += fabs(M[i].y - cM.y);
    }

    if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
        fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
        return 0;
    sm.x = count/sm.x; sm.y = count/sm.y;
    sM.x = count/sM.x; sM.y = count/sM.y;

    double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
    double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
    CvMat _invHnorm = cvMat( 3, 3, CV_64FC1, invHnorm );
    CvMat _Hnorm2 = cvMat( 3, 3, CV_64FC1, Hnorm2 );

    cvZero( &_LtL );
    for( i = 0; i < count; i++ )
    {
        double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
        double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
        double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
        double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
        int j, k;
        for( j = 0; j < 9; j++ )
            for( k = j; k < 9; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
    }
    cvCompleteSymm( &_LtL );

    //cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    cvEigenVV( &_LtL, &matV, &matW );
    cvMatMul( &_invHnorm, &_H0, &_Htemp );
    cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
    cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

    return 1;
}


void HomographyEstimator::computeReprojError( const CvMat* m1, const CvMat* m2,
                                                const CvMat* model, CvMat* _err )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    const double* H = model->data.db;
    float* err = _err->data.fl;

    for( i = 0; i < count; i++ )
    {
        double ww = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double dx = (H[0]*M[i].x + H[1]*M[i].y + H[2])*ww - m[i].x;
        double dy = (H[3]*M[i].x + H[4]*M[i].y + H[5])*ww - m[i].y;
        err[i] = (float)(dx*dx + dy*dy);
    }
}

bool HomographyEstimator::refine( const CvMat* m1, const CvMat* m2, CvMat* model, int maxIters )
{
    CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
    int i, j, k, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    CvMat modelPart = cvMat( solver.param->rows, solver.param->cols, model->type, model->data.ptr );
    cvCopy( &modelPart, solver.param );

    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;

        if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
            break;

        for( i = 0; i < count; i++ )
        {
            const double* h = _param->data.db;
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            double err[] = { _xi - m[i].x, _yi - m[i].y };
            if( _JtJ || _JtErr )
            {
                double J[][8] =
                {
                    { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                    { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                };

                for( j = 0; j < 8; j++ )
                {
                    for( k = j; k < 8; k++ )
                        _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                    _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                }
            }
            if( _errNorm )
                *_errNorm += err[0]*err[0] + err[1]*err[1];
        }
    }

    cvCopy( solver.param, &modelPart );
    return true;
}


CV_IMPL void
cvEstimateHomography( const CvMat* objectPoints, const CvMat* imagePoints,
    CvMat* __H, int method, double ransacReprojThreshold, int maxIters,
    CvMat* mask, CvFundamentalResult *result )
{
    const double confidence = 0.995;
    const double defaultRANSACReprojThreshold = 3;
    bool retval = false;
    Ptr<CvMat> m, M, tempMask;

    // Set some reasonable default values for the 7- and 8-points cases
    result->max_iters = false;
    result->num_iters = 0;

    double H[9];
    CvMat matH = cvMat( 3, 3, CV_64FC1, H );
    int count;

    CV_Assert( CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints) );

    count = MAX(imagePoints->cols, imagePoints->rows);
    CV_Assert( count >= 4 );
    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    m = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( imagePoints, m );

    M = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( objectPoints, M );

    if( mask )
    {
        CV_Assert( CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
            (mask->rows == 1 || mask->cols == 1) &&
            mask->rows*mask->cols == count );
    }
    if( mask || count > 4 )
        tempMask = cvCreateMat( 1, count, CV_8U );
    if( !tempMask.empty() )
        cvSet( tempMask, cvScalarAll(1.) );

    HomographyEstimator estimator(4, maxIters);
    if( count == 4 )
        method = 0;
    if( method == CV_LMEDS )
        retval = estimator.runLMeDS( M, m, &matH, tempMask, confidence );
    else if( method == CV_RANSAC )
        retval = estimator.runRANSAC( M, m, &matH, tempMask, result->num_iters, ransacReprojThreshold, confidence );
    else
        retval = estimator.runKernel( M, m, &matH ) > 0;

    if( retval && count > 4 )
    {
        icvCompressPoints( (CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count );
        count = icvCompressPoints( (CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count );
        M->cols = m->cols = count;
        if( method == CV_RANSAC )
            estimator.runKernel( M, m, &matH );
        estimator.refine( M, m, &matH, 10 );
    }

    if( retval )
        cvConvert( &matH, __H );

    if( mask && tempMask )
    {
        if( CV_ARE_SIZES_EQ(mask, tempMask) )
           cvCopy( tempMask, mask );
        else
           cvTranspose( tempMask, mask );
    }

    result->max_iters = (result->num_iters == maxIters ? true : false);
    result->retval = retval;
    return;
}

// We check whether three correspondences for the homography estimation
// are geometrically consistent (the points in the source image should 
// maintain the same circular order than in the destination image).
//
// The usefullness of this constraint is explained in the paper:
//
// "Speeding-up homography estimation in mobile devices"
// Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
// Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
bool
HomographyEstimator::weakConstraint ( const CvMat* srcPoints, const CvMat* dstPoints, int t1, int t2, int t3 )
{
  const CvPoint2D64f* src = (const CvPoint2D64f*)srcPoints->data.ptr;
  const CvPoint2D64f* dst = (const CvPoint2D64f*)dstPoints->data.ptr;

  CvMat* A = cvCreateMat( 3, 3, CV_64F );
  CvMat* B = cvCreateMat( 3, 3, CV_64F );

  double detA;
  double detB;

  cvmSet(A, 0, 0, src[t1].x);
  cvmSet(A, 0, 1, src[t1].y);
  cvmSet(A, 0, 2, 1);
  cvmSet(A, 1, 0, src[t2].x);
  cvmSet(A, 1, 1, src[t2].y);
  cvmSet(A, 1, 2, 1);
  cvmSet(A, 2, 0, src[t3].x);
  cvmSet(A, 2, 1, src[t3].y);
  cvmSet(A, 2, 2, 1);

  cvmSet(B, 0, 0, dst[t1].x);
  cvmSet(B, 0, 1, dst[t1].y);
  cvmSet(B, 0, 2, 1);
  cvmSet(B, 1, 0, dst[t2].x);
  cvmSet(B, 1, 1, dst[t2].y);
  cvmSet(B, 1, 2, 1);
  cvmSet(B, 2, 0, dst[t3].x);
  cvmSet(B, 2, 1, dst[t3].y);
  cvmSet(B, 2, 2, 1);

  detA = cvDet(A);
  detB = cvDet(B);

  cvReleaseMat(&A);
  cvReleaseMat(&B);

  return (detA*detB >= 0);
};

// We check whether the minimal set of points for the homography estimation
// are geometrically consistent. We check if every 3 correspondences sets
// fulfills the constraint.
//
// The usefullness of this constraint is explained in the paper:
//
// "Speeding-up homography estimation in mobile devices"
// Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
// Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
bool
HomographyEstimator::isMinimalSetConsistent ( const CvMat* srcPoints, const CvMat* dstPoints )
{
  return weakConstraint(srcPoints, dstPoints, 0, 1, 2) &&
         weakConstraint(srcPoints, dstPoints, 1, 2, 3) &&
         weakConstraint(srcPoints, dstPoints, 0, 2, 3) &&
         weakConstraint(srcPoints, dstPoints, 0, 1, 3);
}

/* End of file. */
