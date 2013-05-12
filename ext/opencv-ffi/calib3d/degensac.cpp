
#include "cvffi_modelest.h"
#include "cvffi_fundam.h"

using namespace cv;

/************************************** 7-point algorithm *******************************/
class DegensacEstimator : public FundamentalEstimator
{
  public:
    DegensacEstimator( int _max_iters = 0 );
  protected:

};

DegensacEstimator::DegensacEstimator( int _max_iters )
    : FundamentalEstimator( 7, _max_iters )
{
}

