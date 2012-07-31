/* This code originates in opencv/modules/features/src/featureselect.cpp
 * but has been modified to return the response variable along with
 * the keypoint locations.
 */
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

#include <cstdio>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../keypoint.h"
#include "harris_with_response.h"

namespace cv
{

  template<typename T> struct greaterThanPtr
  {
    bool operator()(const T* a, const T* b) const { return *a > *b; }
  };

}

HarrisKeypoint::HarrisKeypoint()
  : x(0.0), y(0.0), response(0.0)
{;}

HarrisKeypoint::HarrisKeypoint( float _x, float _y, float _response )
  : x(_x), y(_y), response(_response)
{;}

using namespace cv;

void cv::goodFeaturesWithResponse( InputArray _image, 
    std::vector<HarrisKeypoint> &corners,
    InputArray _mask, const HarrisParams_t &params )
{
  Mat image = _image.getMat(), mask = _mask.getMat();

  CV_Assert( params.quality_level > 0 && 
             params.min_distance >= 0 );
  CV_Assert( mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

  Mat eig;
  if( params.use_harris )
    cornerHarris( image, eig, params.block_size, 3, params.harris_k );
  else
    cornerMinEigenVal( image, eig, params.block_size, 3 );

  featuresWithResponseCommon( eig, corners, mask, params );
}


void cv::featuresWithResponseCommon( Mat &eig,
    std::vector<HarrisKeypoint> &corners,
    Mat &mask, const HarrisParams_t &params )
{
  Mat tmp;
  double maxVal = 0;
  minMaxLoc( eig, 0, &maxVal, 0, 0, mask );
  threshold( eig, eig, maxVal*params.quality_level, 0, THRESH_TOZERO );
  dilate( eig, tmp, Mat());

  Size imgsize = eig.size();

  std::vector<const float*> tmpCorners;

  // collect list of pointers into eig - put them into temporary image
  for( int y = 1; y < imgsize.height - 1; y++ )
  {
    const float* eig_data = (const float*)eig.ptr(y);
    const float* tmp_data = (const float*)tmp.ptr(y);
    const uchar* mask_data = mask.data ? mask.ptr(y) : 0;

    for( int x = 1; x < imgsize.width - 1; x++ )
    {
      float val = eig_data[x];
      if( val != 0 && val == tmp_data[x] && (!mask_data || mask_data[x]) )
        tmpCorners.push_back(eig_data + x);
    }
  }

  sort( tmpCorners, greaterThanPtr<float>() );
  size_t i, j, total = tmpCorners.size(), ncorners = 0;

  if(params.min_distance >= 1)
  {
    // Partition the image into larger grids
    int w = eig.cols;
    int h = eig.rows;

    const int cell_size = cvRound(params.min_distance);
    const int grid_width = (w + cell_size - 1) / cell_size;
    const int grid_height = (h + cell_size - 1) / cell_size;

    std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

    float minDistance = params.min_distance;

    for( i = 0; i < total; i++ )
    {
      int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
      int y = (int)(ofs / eig.step);
      int x = (int)((ofs - y*eig.step)/sizeof(float));
      float response = (float)(*(tmpCorners[i]));

      bool good = true;

      int x_cell = x / cell_size;
      int y_cell = y / cell_size;

      int x1 = x_cell - 1;
      int y1 = y_cell - 1;
      int x2 = x_cell + 1;
      int y2 = y_cell + 1;

      // boundary check
      x1 = std::max(0, x1);
      y1 = std::max(0, y1);
      x2 = std::min(grid_width-1, x2);
      y2 = std::min(grid_height-1, y2);

      for( int yy = y1; yy <= y2; yy++ )
      {
        for( int xx = x1; xx <= x2; xx++ )
        {   
          std::vector <Point2f> &m = grid[yy*grid_width + xx];

          if( m.size() )
          {
            for(j = 0; j < m.size(); j++)
            {
              float dx = x - m[j].x;
              float dy = y - m[j].y;

              if( dx*dx + dy*dy < minDistance )
              {
                good = false;
                goto break_out;
              }
            }
          }                
        }
      }

break_out:

      if(good)
      {
        // printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
        //    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
        grid[y_cell*grid_width + x_cell].push_back(Point2f((float)x, (float)y));

        corners.push_back(HarrisKeypoint((float)x, (float)y, response));
        ++ncorners;

        if( params.max_corners > 0 && (int)ncorners == params.max_corners )
          break;
      }
    }
  }
  else
  {
    for( i = 0; i < total; i++ )
    {
      int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
      int y = (int)(ofs / eig.step);
      int x = (int)((ofs - y*eig.step)/sizeof(float));

      corners.push_back(HarrisKeypoint((float)x, (float)y, 
                                       (float)(*(tmpCorners[i]))));
      ++ncorners;
      if( params.max_corners > 0 && (int)ncorners == params.max_corners )
        break;
    }
  }

  /*
     for( i = 0; i < total; i++ )
     {
     int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
     int y = (int)(ofs / eig.step);
     int x = (int)((ofs - y*eig.step)/sizeof(float));

     if( minDistance > 0 )
     {
     for( j = 0; j < ncorners; j++ )
     {
     float dx = x - corners[j].x;
     float dy = y - corners[j].y;
     if( dx*dx + dy*dy < minDistance )
                    break;
            }
            if( j < ncorners )
                continue;
        }

        corners.push_back(Point2f((float)x, (float)y));
        ++ncorners;
        if( maxCorners > 0 && (int)ncorners == maxCorners )
            break;
    }
*/
}

extern "C" 
  CvSeq *cvGoodFeaturesWithResponse( const void* _image, 
      const void* _maskImage, 
      CvMemStorage *pool, const HarrisParams_t params )
{
  cv::Mat image = cv::cvarrToMat(_image), mask;
  std::vector<HarrisKeypoint> corners;

  if( _maskImage )
    mask = cv::cvarrToMat(_maskImage);

  goodFeaturesWithResponse( image, corners, mask, params );

  CvSeqWriter writer;
  cvStartWriteSeq( 0, sizeof(CvSeq), sizeof(CvKeyPoint_t), pool, &writer );
  size_t ncorners = corners.size();
  for( size_t i = 0; i < ncorners; i++ ) {
    CvKeyPoint_t kp;
    kp.x = corners[i].x;
    kp.y = corners[i].y;
    kp.response = corners[i].response;

    kp.size = kp.angle = 0.0;
    kp.octave = 0;

    CV_WRITE_SEQ_ELEM( kp, writer );
  }

  return cvEndWriteSeq( &writer );
}

/* End of file. */
