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

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/core/core_c.h>

#define CV_DEGENSAC 9

#ifndef _CVFFI_FUNDAM_H
#define _CVFFI_FUNDAM_H

namespace cv {
  /** Snipped out the homography estimator.  -AMM **/

  /* Evaluation of Fundamental Matrix from point correspondences.
     The original code has been written by Valery Mosyagin */

  /* The algorithms (except for RANSAC) and the notation have been taken from
     Zhengyou Zhang's research report
     "Determining the Epipolar Geometry and its Uncertainty: A Review"
     that can be found at http://www-sop.inria.fr/robotvis/personnel/zzhang/zzhang-eng.html */

  /* This version modified by Aaron Marburg */

  /************************************** 7-point algorithm *******************************/
  class FundamentalEstimator : public CvffiModelEstimator2
  {
    public:
      FundamentalEstimator( int _modelPoints, int _max_iters = 0 );

      virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
      virtual int run7Point( const CvMat* m1, const CvMat* m2, CvMat* model );
      virtual int run8Point( const CvMat* m1, const CvMat* m2, CvMat* model );
    protected:
      virtual void computeReprojError( const CvMat* m1, const CvMat* m2,
          const CvMat* model, CvMat* error );

  };

}

extern "C" {
  struct CvFundamentalResult {
    int retval;
    int num_iters;
    bool max_iters;
  };


  /* Main C entry point */
  CV_IMPL void cvEstimateFundamental( const CvMat* points1, const CvMat* points2,
      CvMat* fmatrix, int method,
      double param1, double param2, int max_iters,  CvMat* mask,
      CvFundamentalResult *result );

}

#endif /* _CVFFI_FUNDAM_H */
