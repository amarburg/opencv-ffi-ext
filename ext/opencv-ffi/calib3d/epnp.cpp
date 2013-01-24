/* C wrapper code to allow access to the solvePnPRansac and solvePnP
 * functions */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core_c.h>

using namespace cv;

extern "C" {

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
    Mat _objectPts( objectPoints ),
    _imagePts( imagePoints ),
    _cameraMat( cameraMatrix ),
    _distCoeffs( distCoeffs );

    Mat _rvec( rvec ), _tvec( tvec ), _inliers;

    solvePnPRansac( _objectPts, _imagePts, _cameraMat, _distCoeffs,
        _rvec, _tvec, useExtrinsicGuess, iterationsCount, reprojectionError,
        minInliersCount, _inliers, flags );

    *rvec = _rvec;
    *tvec = _tvec;
    if( inliers != NULL )
      *inliers = _inliers;
  }
}
