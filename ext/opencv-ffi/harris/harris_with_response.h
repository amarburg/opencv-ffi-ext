
#ifndef _NEW_HARRIS_H_
#define _NEW_HARRIS_H_

#include <vector>

typedef struct HarrisParams_t {
  double quality_level;
  double min_distance;
  int block_size;
  double harris_k;
  int use_harris;
  int max_corners;
} NewHarrisParams_t;


#ifdef __cplusplus

class HarrisKeypoint {
  public:
  
  float x,y,response;

  HarrisKeypoint();
   HarrisKeypoint( float _x, float _y, float _response );
};

namespace cv {

  void goodFeaturesWithResponse( InputArray _image, 
      std::vector<HarrisKeypoint> &corners,
      InputArray _mask, const HarrisParams_t &params );

  void featuresWithResponseCommon( Mat &eig,
      std::vector<HarrisKeypoint> &corners,
      Mat &mask, const HarrisParams_t &params );


}

#endif


#endif
