
#ifndef _NEW_HARRIS_H_
#define _NEW_HARRIS_H_

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

#endif


#endif
