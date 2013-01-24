/* C wrapper code to allow access to the solvePnPRansac and solvePnP
 * functions */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core_c.h>

#include <iostream>
using namespace std;

using namespace cv;

extern "C" {

  void cvSolvePnP( CvMat *objectPoints, CvMat *imagePoints,
      CvMat *cameraMatrix, CvMat *distCoeffs,
      CvMat *rvec, CvMat *tvec, 
      bool useExtrinsicGuess CV_DEFAULT(false),
      int flags CV_DEFAULT(CV_ITERATIVE) )
  {
    // TODO:  Should check that objectPoints and imagePoints
    // are of correct dimensions and equal to each other...
    // or let solvePnPRansac do it...

    Mat _objectPts = cvarrToMat( objectPoints ),
    _imagePts = cvarrToMat( imagePoints ),
    _cameraMat = cvarrToMat( cameraMatrix ),
    _distCoeffs = cvarrToMat( distCoeffs );

    CV_Assert( CV_MAT_TYPE(rvec->type) == CV_64F && 
        CV_MAT_TYPE(tvec->type) == CV_64F &&
        rvec->rows == 3 && rvec->cols == 1 &&
        tvec->rows == 3 && tvec->cols == 1 );

    Mat _rvec = cvarrToMat( rvec ), _rvec0 = rvec, 
        _tvec = cvarrToMat( tvec ), _tvec0 = tvec;

    solvePnP( _objectPts, _imagePts, _cameraMat, _distCoeffs,
        _rvec, _tvec, useExtrinsicGuess, flags );

    CV_Assert( _tvec0.data == _tvec.data );
    CV_Assert( _rvec0.data == _rvec.data );
  }


  void cvSolvePnPRansac( CvMat *objectPoints, CvMat *imagePoints,
      CvMat *cameraMatrix, CvMat *distCoeffs,
      CvMat *rvec, CvMat *tvec, 
      bool useExtrinsicGuess CV_DEFAULT(false),
      int iterationsCount CV_DEFAULT(100),
      float reprojectionError CV_DEFAULT(8.0),
      int minInliersCount CV_DEFAULT(100),
      CvMat *inliers CV_DEFAULT(NULL),
      int flags CV_DEFAULT(CV_ITERATIVE) )
  {
    // TODO:  Should check that objectPoints and imagePoints
    // are of correct dimensions and equal to each other...
    // or let solvePnPRansac do it...

    Mat _objectPts = cvarrToMat( objectPoints ),
    _imagePts = cvarrToMat( imagePoints ),
    _cameraMat = cvarrToMat( cameraMatrix ),
    _distCoeffs = cvarrToMat( distCoeffs );

    CV_Assert( CV_MAT_TYPE(rvec->type) == CV_64F && 
        CV_MAT_TYPE(tvec->type) == CV_64F &&
        rvec->rows == 3 && rvec->cols == 1 &&
        tvec->rows == 3 && tvec->cols == 1 );

    Mat _rvec = cvarrToMat( rvec ), _rvec0 = rvec, 
        _tvec = cvarrToMat( tvec ), _tvec0 = tvec,
        _inliersDst;

    if( inliers != NULL ) {
      int count = MAX( objectPoints->cols, objectPoints->rows );
      CV_Assert( CV_IS_MASK_ARR(inliers) && CV_IS_MAT_CONT(inliers->type) &&
          (inliers->rows == 1 || inliers->cols == 1) &&
          inliers->rows*inliers->cols == count );
    }

    solvePnPRansac( _objectPts, _imagePts, _cameraMat, _distCoeffs,
        _rvec, _tvec, useExtrinsicGuess, iterationsCount, reprojectionError,
        minInliersCount, _inliersDst, flags );

    CV_Assert( _tvec0.data == _tvec.data );
    CV_Assert( _rvec0.data == _rvec.data );

    if( inliers != NULL ) {
      cvSet( inliers, cvScalarAll(0.0) );
      for( int i = 0; i < _inliersDst.rows; i++ ) {
        cvSet1D( inliers, _inliersDst.at<int>(i), cvScalarAll(1.) );
      }
    }
  }
}
