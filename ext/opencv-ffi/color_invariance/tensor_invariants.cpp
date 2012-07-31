
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
    HS_QUASI_INVARIANTS, GREYSCALE };

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

    CV_Assert( src.channels() == 3 );
    CV_Assert( src.depth() == CV_32F );

    if( which == HS_QUASI_INVARIANTS ) {
      dst.create( src.size(), CV_32FC4 );
    } else {
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

      //Sobel( e_mat, ex_mat, -1, 1, 0, CV_SCHARR );
      //Sobel( e_mat, ey_mat, -1, 0, 1, CV_SCHARR );

      for( int i = 0; i < sz.height; i++ ) {
        for( int j = 0; j < sz.width; j++ ) {
          Vec2f out;
          out[0] = dx_mat.at<float>(i,j);
          out[1] = dy_mat.at<float>(i,j);
          dst.at<Vec2f>(i,j) = out;
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
              if( (i == 1847) && (j==186) ) {
                cout << "Hx,Hy,Sx,Sy " << i << "," << j << " ==> " << out[0] << "," << out[1] << endl;
              }

              break;
            case S_QUASI_INVARIANT:
              sdenom = (e*e + el*el + ell*ell) * sqrt( el*el+ell*ell );
              out[0] = ( e * (el*elx +ell*ellx) - ex * (el*el + ell*ell))/ sdenom;
              out[1] = ( e * (el*ely +ell*elly) - ey * (el*el + ell*ell))/ sdenom;
              dst.at<Vec2f>(i,j) = out;
              break;
          }

          if( (i == 1847) && (j==186) ) {
            cout << "e   " << i << "," << j << " ==> " << e_p[0] << "," << e_p[1] << "," << e_p[2] <<  endl;
            cout << "ex  " << i << "," << j << " ==> " << ex_p[0] << "," << ex_p[1] << "," << ex_p[2] << endl;
            cout << "ey  " << i << "," << j << " ==> " << ey_p[0] << "," << ey_p[1] << "," << ey_p[2] << endl;
            cout << "Hx,Hy,Sx,Sy " << i << "," << j << " ==> " << outt[0] << "," << outt[1] << "," << outt[2] << "," << outt[3] << endl;
          }

        }
      }
    }
  }


  static void generateImageTensor( const Mat &src, Mat &dst, int blockSize = 3 )
  {
    Size size = src.size();

    // Create's M, which is equivalent to Harris' image tensor
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

        if( (i == 1847) && (j == 186) ) {
          cout << "k = " << k << endl;
          cout << "At " << i << "," << j << " xx = " << xx << " xy = " << xy << " yy = " << yy << endl;
          cout << "At " << i << "," << j << " I = " << _dst.at<float>(i,j) << endl;
        }
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

  template<typename T> struct greaterThanPtr
  {
    bool operator()(const T* a, const T* b) const { return *a > *b; }
  };

  // Contains code common to the Harris-style functions after the computation
  // of the response variable.
  static void harrisCommon( Mat &eig,
      OutputArray _corners,
      int maxCorners, double qualityLevel, double minDistance,
      const Mat &mask,
      bool useHarrisDetector,
      double harrisK )
  {
    Mat tmp;
    double maxVal = 0, minVal = 0;
    minMaxLoc( eig, &minVal, &maxVal, 0, 0, mask );
    cout << "Found maximum value " << maxVal << endl;
    cout << "Found minimum value " << minVal << endl;
    cout << "Quality level " << qualityLevel<< endl;

    threshold( eig, eig, maxVal*qualityLevel, 0, THRESH_TOZERO );
    dilate( eig, tmp, Mat());

    Size imgsize = eig.size();

    vector<const float*> tmpCorners;

    // collect list of pointers to features - put them into temporary image
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
    vector<Point2f> corners;
    size_t i, j, total = tmpCorners.size(), ncorners = 0;

    if(minDistance >= 1)
    {
      // Partition the image into larger grids
      int w = eig.cols;
      int h = eig.rows;

      const int cell_size = cvRound(minDistance);
      const int grid_width = (w + cell_size - 1) / cell_size;
      const int grid_height = (h + cell_size - 1) / cell_size;

      std::vector<std::vector<Point2f> > grid(grid_width*grid_height);

      minDistance *= minDistance;

      for( i = 0; i < total; i++ )
      {
        int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
        int y = (int)(ofs / eig.step);
        int x = (int)((ofs - y*eig.step)/sizeof(float));

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
            vector <Point2f> &m = grid[yy*grid_width + xx];

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

          corners.push_back(Point2f((float)x, (float)y));
          ++ncorners;

          if( maxCorners > 0 && (int)ncorners == maxCorners )
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

        corners.push_back(Point2f((float)x, (float)y));
        ++ncorners;
        if( maxCorners > 0 && (int)ncorners == maxCorners )
          break;
      }
    }

    Mat(corners).convertTo(_corners, _corners.fixedType() ? _corners.type() : CV_32F);

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

  void quasiInvariantFeaturesToTrack( QuasiInvariant which,
      const Mat &img,
      OutputArray _corners,
      int maxCorners, double qualityLevel, 
      double minDistance, int blockSize,
      const Mat &mask, 
      bool useHarrisDetector,
      double harrisK )
  {
    Size size = img.size();
    Mat qi;
    Mat m_mat( img.size(), CV_32FC3 );

    generateQuasiInvariants( img, qi, which );

    vector<Mat> channels;
    split( qi, channels );
    for(int i = 0; i < qi.channels(); i++ ) {
      double minVal, maxVal;
      Mat chan = channels[i];
      minMaxLoc( chan, &minVal, &maxVal );
      cout << "For channel " << i << " max is " << maxVal << " ; min is " << minVal << endl;
    }

    generateImageTensor( qi, m_mat, blockSize );
    
    // Stores the resulting Harris I or minimum Eigenvalues
    Mat eig;

    if( useHarrisDetector )
      quasiInvariantHarris( m_mat, eig, harrisK );
    else
      quasiInvariantMinEigen( m_mat, eig );

    harrisCommon( eig, _corners,
        maxCorners, qualityLevel, minDistance,
        mask, useHarrisDetector, harrisK );
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
    Mat dst = cvarrToMat( dstarr );

    // Cast back to the type of dstarr
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
  }

  void cvNormalizedColorImage( CvMat *srcarr, CvMat *dstarr )
  {
    Mat src, dst;
    convertCvMatToMat( srcarr, src ); 
    cv::normalizedColorImage( src, dst );
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

  void cvQuasiInvariantFeaturesToTrack( QuasiInvariant which,
      const CvMat *_chu,
      CvPoint2D32f* _corners, int *_corner_count,
      double quality_level, double min_distance,
      int block_size,
      const void* _maskImage, 
      int use_harris, double harris_k )
  {
    Mat chu, mask;
    cv::vector<cv::Point2f> corners;

    convertCvMatToMat( _chu, chu );

    if( _maskImage )
      mask = cv::cvarrToMat(_maskImage);

    CV_Assert( _corners && _corner_count );
    cv::quasiInvariantFeaturesToTrack( which, chu, corners, *_corner_count, quality_level,
        min_distance, block_size, mask, use_harris != 0, harris_k );

    size_t i, ncorners = corners.size();
    for( i = 0; i < ncorners; i++ ) {
      _corners[i] = corners[i];
    }
    *_corner_count = (int)ncorners;
  }

}

