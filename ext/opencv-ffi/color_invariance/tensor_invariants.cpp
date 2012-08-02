
#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../harris/harris_with_response.h"
#include "../keypoint.h"

#include <iostream>
#include <stdio.h>

using namespace std;

#define KSIZE 1

#define Pixel Vec3f

namespace cv {

  double sdot( const Pixel &a, const Pixel &b )
  {
    return a(0)*b(0)+ a(1)*b(1)+ a(2)*b(2);
  }

  void normalizedColorImage( Mat &src, Mat &dst )
  {
    Size sz = src.size();

    CV_Assert( src.channels() == 3 );
    CV_Assert( src.depth() == CV_32F );

    dst.create( sz, CV_MAKETYPE( CV_32F, 3 ) );

    for( int i = 0; i < sz.height; i++ ) {
      for( int j = 0; j < sz.width; j++ ) {
        Pixel fhat = src.at<Pixel>( i,j );
        float fnorm = 1.0/norm(fhat);
        fhat *= fnorm;

        dst.at<Pixel>(i,j) = fhat;

        if( i == 0 and j == 0) {
          Pixel p = dst.at<Pixel>(i,j);
          cout << "At " << i << "," << j << " = " << p[0] << "," << p[1] << "," << p[2] << endl;
        }

      }
    }
  }


  

  enum QuasiInvariant { H_QUASI_INVARIANT = 0, 
                        S_QUASI_INVARIANT, 
                        HS_QUASI_INVARIANTS, 
                        RGB_GRADIENT,
                        GREYSCALE };

  // Generates the Hx, Hy, Sx, and Sy color invariants from Geusebroek et al
  // "Color Invariance"
  // DOI: ...
  //
  // and Chu et al 
  // "Color-based corner detection by color invariance."
  // DOI: 10.1109/HAVE.2011.6088390
  //
  //
  // Stores them in a 4-channel mat in this order (Hx, Hy, Sx, Sy)
  void generateQuasiInvariants( const Mat &src, Mat &dst, QuasiInvariant which,
      int aperture_size = 3, int block_size = 3 )
  {
    Size sz = src.size();

    // This is fixed in the OpenCV implementation of GoodFeaturesToTrack
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if( aperture_size < 0 )
      scale *= 2.;
    if( src.depth() == CV_8U )
      scale *= 255.;
    scale = 1/scale;

    // Function only works on 3-channel BGR 32F images
    CV_Assert( src.channels() == 3 );
    CV_Assert( src.depth() == CV_32F );

    if( which == HS_QUASI_INVARIANTS ) {
      dst.create( src.size(), CV_32FC4 );
    } else if ( which == RGB_GRADIENT ) {
      dst.create( src.size(), CV_32FC3 );
    }else {
      dst.create( src.size(), CV_32FC2 );
    }

    CV_Assert( src.depth() == dst.depth() );
    //cout << "Size is " << sz.width << "x" << sz.height << endl;

    if( which == GREYSCALE ) {

      Mat g_mat( src.size(), CV_32FC1 );

      for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Pixel px = src.at<Pixel>( i,j );

          // TODO:  Doesn't correct for other channel orderings
          float b = px[0], g = px[1], r = px[2];

          g_mat.at<float>(i,j) = 0.299*r +  0.587*g + 0.114*b;
        }
      }

      Mat dx_mat( g_mat.size(), g_mat.type() );
      Mat dy_mat( g_mat.size(), g_mat.type() );

      if( aperture_size > 0 )
      {
        Sobel( g_mat, dx_mat, CV_32F, 1, 0, aperture_size, scale, 0 ); //, borderType );
        Sobel( g_mat, dy_mat, CV_32F, 0, 1, aperture_size, scale, 0 ); //, borderType );
      }
      else
      {
        Scharr( g_mat, dx_mat, CV_32F, 1, 0, scale, 0 ); //, borderType );
        Scharr( g_mat, dy_mat, CV_32F, 0, 1, scale, 0 ); //, borderType );
      }

      for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Vec2f out;
          out[0] = dx_mat.at<float>(i,j);
          out[1] = dy_mat.at<float>(i,j);
          dst.at<Vec2f>(i,j) = out;
       }
      }

    } else if ( which == RGB_GRADIENT ) {
 
      Mat dx_mat( src.size(), CV_32FC3 );
      Mat dy_mat( src.size(), CV_32FC3 );

      vector<Mat> chans, dxs, dys;

      split( src, chans );

      for( unsigned int i = 0; i < 3; i++ ) {
        Mat dx_foo( src.size(), CV_32FC1 );
        Mat dy_bar( src.size(), CV_32FC1 );

        if( aperture_size > 0 )
        {
          Sobel( chans[i], dx_foo, CV_32F, 1, 0, aperture_size, scale, 0 ); //, borderType );
          Sobel( chans[i], dy_bar, CV_32F, 0, 1, aperture_size, scale, 0 ); //, borderType );
        }
        else
        {
          Scharr( chans[i], dx_foo, CV_32F, 1, 0, scale, 0 ); //, borderType );
          Scharr( chans[i], dy_bar, CV_32F, 0, 1, scale, 0 ); //, borderType );
        }    
        dxs.push_back( dx_foo );
        dys.push_back( dy_bar );
      }

      merge( dxs, dx_mat );
      merge( dys, dy_mat );

       for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Vec3f out; //  f_x^T f_x, f_x^T f_y, f_y^T f^y
          Vec3f px( dx_mat.at<float>(i,j) );
          Vec3f py( dy_mat.at<float>(i,j) );

          out[0] = px[0]*px[0] + px[1]*px[1] + px[2]*px[2];
          out[1] = px[0]*py[0] + px[1]*py[1] + px[2]*py[2];
          out[2] = py[0]*py[0] + py[1]*py[1] + py[2]*py[2];
          dst.at<Vec3f>(i,j) = out;
       }
      }
     
    } else {

      Mat e_mat( src.size(), CV_32FC3 );

      for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Pixel px = src.at<Pixel>( i,j );

          // TODO:  Doesn't correct for other channel orderings
          float b = px[0], g = px[1], r = px[2];

          float e   = 0.06 * r + 0.63 * g + 0.27 * b;
          float el  =  0.3 * r + 0.04 *g  - 0.35 * b;
          float ell = 0.34 * r - 0.6 * g  + 0.17 * b;

          Pixel out;
          out[0] = e; out[1] = el; out[2] = ell;
          e_mat.at<Pixel>(i,j) = out;

          //if( (i == 100) && (j==100) ) {
          //      cout << "Original  r: " << r << "  g: " << g << " b: " << b << endl;
          //      }

        }
      }

      Mat ex_mat( e_mat.size(), e_mat.type() );
      Mat ey_mat( e_mat.size(), e_mat.type() );

      if( aperture_size > 0 )
      {
        Sobel( e_mat, ex_mat, CV_32F, 1, 0, aperture_size, scale, 0 ); //, borderType );
        Sobel( e_mat, ey_mat, CV_32F, 0, 1, aperture_size, scale, 0 ); //, borderType );
      }
      else
      {
        Scharr( e_mat, ex_mat, CV_32F, 1, 0, scale, 0 ); //, borderType );
        Scharr( e_mat, ey_mat, CV_32F, 0, 1, scale, 0 ); //, borderType );
      }

      //Sobel( e_mat, ex_mat, -1, 1, 0, CV_SCHARR );
      //Sobel( e_mat, ey_mat, -1, 0, 1, CV_SCHARR );

      for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Pixel e_p  = e_mat.at<Pixel>(i,j);
          Pixel ex_p = ex_mat.at<Pixel>(i,j);
          Pixel ey_p = ey_mat.at<Pixel>(i,j);

          float e = e_p[0], el = e_p[1], ell = e_p[2];
          float ex = ex_p[0], elx = ex_p[1], ellx = ex_p[2];
          float ey = ey_p[0], ely = ey_p[1], elly = ey_p[2];

          Vec2f out;
          Vec4f outt;
          float hdenom, sdenom;

          // TODO:  Fix the DRY here.
          switch( which )  {
            case HS_QUASI_INVARIANTS:
              hdenom = el*el + ell*ell;
              outt[0] = (ell * elx - el * ellx )/hdenom;
              outt[1] = (ell * ely - el * elly )/hdenom;
              sdenom = (e*e + el*el + ell*ell) * sqrt( el*el+ell*ell );
              outt[2] = ( e * (el*elx + ell*ellx) - ex * (el*el + ell*ell))/ sdenom;
              outt[3] = ( e * (el*ely + ell*elly) - ey * (el*el + ell*ell))/ sdenom;
              dst.at<Vec4f>(i,j) = outt;
              break;
            case H_QUASI_INVARIANT:
              hdenom = el*el + ell*ell;
              out[0] = (ell * elx - el * ellx )/hdenom;
              out[1] = (ell * ely - el * elly )/hdenom;
              dst.at<Vec2f>(i,j) = out;
              break;
            case S_QUASI_INVARIANT:
              sdenom = (e*e + el*el + ell*ell) * sqrt( el*el+ell*ell );
              out[0] = ( e * (el*elx +ell*ellx) - ex * (el*el + ell*ell))/ sdenom;
              out[1] = ( e * (el*ely +ell*elly) - ey * (el*el + ell*ell))/ sdenom;
              dst.at<Vec2f>(i,j) = out;
              break;
          }

        }
      }
    }
  }

void spatialQuasiInvariantImage( const Mat &src, Mat &dst, QuasiInvariant which, int aperture_size = 3, int block_size = 3, bool do_normalize = true )
  {
    CV_Assert( (which == H_QUASI_INVARIANT) || (which == S_QUASI_INVARIANT) );

    Size sz = src.size();
    Mat d( sz, CV_32FC1 );

    Mat qi;
    generateQuasiInvariants( src, qi, which, aperture_size, block_size );

    for( int i = 0; i < sz.height; i++ ) {
      for( int j = 0; j < sz.width; j++ ) {
        Vec2f px = qi.at<Vec2f>( i,j );

        // TODO:  Doesn't correct for other channel orderings
        float x = px[0], y = px[1];

        d.at<float>(i,j) =  sqrt( x*x + y*y );
      }
    }

    if( do_normalize == true ) {
    // Rescale
      double minVal, maxVal;
      minMaxLoc( d, &minVal, &maxVal );
      cout << "d minimum value = " << minVal << endl;
      cout << "d maximum value = " << maxVal << endl;

      double scale = 1/(maxVal - minVal );

      dst = (d-minVal)*scale;

      minMaxLoc( dst, &minVal, &maxVal );
      cout << "Dst minimum value = " << minVal << endl;
      cout << "Dst maximum value = " << maxVal << endl;

    } else {
      dst = d;
    }
  }



  static void generateImageTensor( const Mat &src, Mat &dst, int blockSize = 3 )
  {
    Size size = src.size();

    // Creates M, which is equivalent to Harris' image tensor
    dst.create( size, CV_32FC3 );

    if( src.channels() == 4 ) {
      // If there are four channels, assume x1, y1, x2, y2
      // For example, for Chu's coefficients, it's [Hx, Hy, Sx, Sy]
      for( int i = 0; i < size.height; i++ ) {
        for( int j = 0; j < size.width; j++ ) {
          Vec4f px = src.at<Vec4f>(i,j);
          Vec3f m;

          // Store the entries of M: xx, xy, yy
          m[0] = px[0]*px[0] + px[2]*px[2];
          m[1] = px[0]*px[1] + px[2]*px[3];
          m[2] = px[1]*px[1] + px[3]*px[3];

          dst.at<Vec3f>(i,j) = m;
        }
      }
    } else if(src.channels() == 3 ) {
      // Pass through
      dst = src;
    } else if(src.channels() == 2) {
      for( int i = 0; i < size.height; i++ ) {
        for( int j = 0; j < size.width; j++ ) {
          Vec2f px = src.at<Vec2f>(i,j);
          Vec3f m;

          // Assume the two channels are x, y
          // Store the entries of M: xx, xy, yy
          m[0] = px[0]*px[0];
          m[1] = px[0]*px[1];
          m[2] = px[1]*px[1];

          dst.at<Vec3f>(i,j) = m;
        }
      }
    }


    cout << "Using block filter of size " << blockSize << endl;
    boxFilter(dst, dst, dst.depth(), Size(blockSize, blockSize),
        Point(-1,-1), false ); //, borderType );

    // Gaussian smooth M
 //   Size ksize( 5,5 );
//GaussianBlur( dst, dst, ksize, 0 );

  }


  static void quasiInvariantHarris( const Mat &m, Mat &_dst, double k )
  {
    _dst.create( m.size(), CV_32F );
    Size size = m.size();

    for( int i = 0; i < size.height; i++ ) {
      for( int j = 0; j < size.width; j++ ) {
        Vec3f _m = m.at<Vec3f>(i,j);
        float xx = _m[0], xy = _m[1], yy = _m[2];

        _dst.at<float>(i,j) = xx*yy - xy*xy - k*(xx+yy)*(xx+yy);
      }
    }

  }

  static void quasiInvariantMinEigen( const Mat &m, Mat &_dst )
  {

    _dst.create( m.size(), CV_32F );
    Size size = m.size();

    for( int i = 0; i < size.height; i++ )
    {
      for( int j = 0; j < size.width; j++ )
      {
        Vec3f px = m.at<Vec3f>(i,j);
        float xx = px[0], xy = px[1], yy = px[2];

        _dst.at<float>(i,j) = (xx+yy) - std::sqrt( (xx-yy)*(xx-yy) + xy*xy );
      }
    }

  }


  void quasiInvariantFeatures( QuasiInvariant which,
      InputArray img,
      std::vector<HarrisKeypoint> &corners,
      InputArray _mask, const HarrisParams_t &params )
  {
    Size size = img.size();
    Mat qi;
    Mat m_mat( img.size(), CV_32FC3 );

    generateQuasiInvariants( img.getMat(), qi, which );

    //vector<Mat> channels;
    //split( qi, channels );
    //for(int i = 0; i < qi.channels(); i++ ) {
    //  double minVal, maxVal;
    //  Mat chan = channels[i];
    //  minMaxLoc( chan, &minVal, &maxVal );
    //  cout << "For channel " << i << " max is " << maxVal << " ; min is " << minVal << endl;
    //}

    generateImageTensor( qi, m_mat, params.block_size );

    Mat eig, mask = _mask.getMat();
    if( params.use_harris )
      quasiInvariantHarris( m_mat, eig, params.harris_k ); //params.block_size, 3, params.harris_k );
    else
      quasiInvariantMinEigen( m_mat, eig ); //, params.block_size, 3 );


    featuresWithResponseCommon( eig, corners, mask, params );
  }



  // Implements the algorithms in section II.c of van de Weijer, Gevers and Smeulders
  // "Robust Photometric Invariant Features From the Color Tensor"
  // DOI:  10.1109/TIP.2005.860343
  //
  void generateColorTensor( Mat &src, Mat &fx, Mat &fy )
  {
    Size sz = src.size();

    CV_Assert( src.channels() == 3 );
    CV_Assert( src.depth() == CV_32F );

    fx.create( sz, CV_MAKETYPE( CV_32F, 3 ) );
    fy.create( sz, CV_MAKETYPE( CV_32F, 3 ) );

    //Ptr<FilterEngine> filter_x = createDerivFilter( src.type(), fx.type(), 1, 0,  KSIZE );
    //Ptr<FilterEngine> filter_y = createDerivFilter( src.type(), fy.type(), 0, 1,  KSIZE );
    //filter_x->apply( src, fx );
    //filter_y->apply( src, fy );

    Sobel( src, fx, CV_32F, 1, 0, CV_SCHARR );
    Sobel( src, fy, CV_32F, 0, 1, CV_SCHARR );
  }

  void generateSQuasiInvariant( Mat &src, Mat &scx, Mat &scy )
  {
    Size sz = src.size();

    CV_Assert( src.channels() == 3 );
    CV_Assert( src.depth() == CV_32F );

    scx.create( sz, CV_MAKETYPE( CV_32F, 3 ) );
    scy.create( sz, CV_MAKETYPE( CV_32F, 3 ) );

    CV_Assert( src.type() == scx.type() );
    CV_Assert( src.type() == scy.type() );

    //cout << "Size is " << sz.width << "x" << sz.height << endl;

    Mat fx( sz, CV_32FC(3) );
    Mat fy( sz, CV_32FC(3) );

    Mat sx( sz, CV_MAKETYPE( CV_32F, 3 ) );
    Mat sy( sz, CV_MAKETYPE( CV_32F, 3 ) );

    //Ptr<FilterEngine> filter_x = createDerivFilter( src.type(), fx.type(), 1, 0,  KSIZE );
    //Ptr<FilterEngine> filter_y = createDerivFilter( src.type(), fy.type(), 0, 1,  KSIZE );
    //filter_x->apply( src, fx );
    //filter_y->apply( src, fy );

    Sobel( src, fx, CV_32F, 1, 0, CV_SCHARR );
    Sobel( src, fy, CV_32F, 0, 1, CV_SCHARR );

    for( int i = 0; i < sz.height; i++ ) {
      for( int j = 0; j < sz.width; j++ ) {
        Pixel fhat = src.at<Pixel>( i,j );
        float fnorm = 1.0/norm(fhat);
        fhat *= fnorm;

        sx.at<Pixel>( i,j ) = fhat * sdot( fx.at<Pixel>(i,j), fhat );
        scx.at<Pixel>(i,j ) = (fx.at<Pixel>(i,j) - sx.at<Pixel>(i,j)); 
        //      if( i == 100 and j == 108 ) {
        //        Pixel s = sx.at<Pixel>(i,j), sc = scx.at<Pixel>(i,j), f = fx.at<Pixel>(i,j);

        //        cout << "fhat " << fhat[0] << "," << fhat[1] << "," << fhat[2] <<endl;
        //        cout << "fx " << f[0] << "," << f[1] << "," << f[2] <<endl;
        //        cout << "sx  " << s[0] << "," << s[1] << "," << s[2] << endl;
        //        cout << "scx " << sc[0] << "," << sc[1] << "," << sc[2] << endl;
        //      }

        sy.at<Pixel>( i,j ) = fhat * sdot( fy.at<Pixel>(i,j), fhat );
        scy.at<Pixel>(i,j ) = (fy.at<Pixel>(i,j) - sy.at<Pixel>(i,j)); 
      }
    }
  }

}

using namespace cv;

extern "C" {
  // The C wrappers

  static void convertCvMatToMat( const CvMat *srcarr, Mat &srcmat )
  {
    Mat src = cvarrToMat(srcarr);

    // Input must be cast to 32FC3
    CV_Assert( src.channels() == 3 );
    switch( src.depth() ) {
      case CV_8U:
        src.convertTo( srcmat, CV_32F, 1.0/255.0, 0 );
        break;
      case CV_32F:
        srcmat = src;
        break;
      default:
        cout << "cvGenerateChuColorInvariants cannot deal with type " << src.depth() << endl;
    }
  }

  static void convertMatToCvMat( Mat &dstmat, CvMat *dstarr )
  {
    Mat dst0 = cvarrToMat( dstarr ), dst = dst0;

//    printf("Before, dst0.data = %p\n", dst0.data );
//    printf("Before, dst.data  = %p\n", dst.data );

    // Cast back to the type of dstarr
    //dst.create( dstmat.size(), dst.type() );
    
    switch( dst.depth() ) {
      case CV_8U:
        dstmat.convertTo( dst, CV_8U, 256.0, 0 );
        break;
      case CV_32F:
        // Strictly speaking, you should be able to set scx.data == dstx
        // but the syntax escapes me...
        dstmat.copyTo( dst );
        break;
      default:
        cout << "cvGenerateChuColorInvariant cannot deal with type " << dstmat.depth() << endl;
    }

//    printf("After, dst0.data = %p\n", dst0.data );
//    printf("After, dst.data  = %p\n", dst.data );

    // This assertion will fire if it's necessary to reallocate the 
    // dstarr->data array.  Make sure the incoming CvMat is the correct type.
    CV_Assert( dst.data == dst0.data );

    //double minVal, maxVal;
    //cvMinMaxLoc( dstarr, &minVal, &maxVal );
    //printf("dstarr minVal = %lf, maxVal = %lf\n", minVal, maxVal );
  }


  void cvNormalizedColorImage( CvMat *srcarr, CvMat *dstarr )
  {
    Mat src, dst;
    convertCvMatToMat( srcarr, src ); 
    cv::normalizedColorImage( src, dst );
    convertMatToCvMat( dst, dstarr );
  }

  void cvSpatialQuasiInvariantImage( QuasiInvariant which, CvMat *srcarr, CvMat *dstarr )
  {
    Mat src, dst;
    convertCvMatToMat( srcarr, src ); 
    cv::spatialQuasiInvariantImage( src, dst, which );
    convertMatToCvMat( dst, dstarr );
  }

  void cvGenerateQuasiInvariant( QuasiInvariant which,
                                 CvMat *srcarr, CvMat *dstarr )
  {
    Mat src, dst;
    convertCvMatToMat( srcarr, src ); 
    cv::generateQuasiInvariants( src, dst, which );
    convertMatToCvMat( dst, dstarr );
  }

  void cvGenerateHQuasiInvariant( CvMat *srcarr, CvMat *dstarr )
  {
    cvGenerateQuasiInvariant( H_QUASI_INVARIANT, srcarr, dstarr );
  }

  void cvGenerateSQuasiInvariant( CvMat *srcarr, CvMat *dstarr )
  {
    cvGenerateQuasiInvariant( S_QUASI_INVARIANT, srcarr, dstarr );
  }

  void cvGenerateChuQuasiInvariants( CvMat *srcarr, CvMat *dstarr )
  {
    cvGenerateQuasiInvariant( HS_QUASI_INVARIANTS, srcarr, dstarr );
  }



  void cvGenerateColorTensor( CvMat *srcarr, CvMat *scx, CvMat *scy )
  {
    Mat src = cvarrToMat(srcarr);
    Mat cvtSrc;

    // Input must be cast to 32FC3
    CV_Assert( src.channels() == 3 );
    switch( src.depth() ) {
      case CV_8U:
        src.convertTo( cvtSrc, CV_32F, 1.0, 0 );
        break;
      case CV_32F:
        cvtSrc = src;
        break;
      default:
        cout << "cvGenerateColorTensor cannot deal with type " << src.depth() << endl;
    }

    Mat scxMat = cvarrToMat(scx), dstx;
    Mat scyMat = cvarrToMat(scy), dsty;

    cv::generateColorTensor( cvtSrc, dstx, dsty );

    // Cast back to scx, scy type
    switch( scxMat.depth() ) {
      case CV_8U:
        dstx.convertTo( scxMat, CV_8U, 256.0, 0 );
        dsty.convertTo( scyMat, CV_8U, 256.0, 0 );
        break;
      case CV_32F:
        // Strictly speaking, you should be able to set scx.data == dstx
        // but the syntax escapes me...
        dstx.copyTo( scxMat );
        dsty.copyTo( scyMat );
        break;
      default:
        cout << "cvGenerateColorTensor cannot deal with type " << scxMat.depth() << endl;
    }
  }

  void cvFoobar( QuasiInvariant which,
      const void* _image, 
      const void* _maskImage, 
      CvMemStorage *pool, const HarrisParams_t params )
  {
  }

  CvSeq *cvQuasiInvariantFeatures( QuasiInvariant which,
      const CvMat* _image, 
      const void* _maskImage, 
      CvMemStorage *pool, const HarrisParams_t params )
  {
    cv::Mat image = cv::cvarrToMat(_image), mask;
    convertCvMatToMat( _image, image );

    std::vector<HarrisKeypoint> corners;

    if( _maskImage )
      mask = cv::cvarrToMat(_maskImage);

    quasiInvariantFeatures( which, image, corners, mask, params );

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

}

