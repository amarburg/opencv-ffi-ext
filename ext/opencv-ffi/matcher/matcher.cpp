#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core_c.h>
#include <vector>

#include <stdio.h>

using namespace cv;

struct CvDMatch_t {
  int queryIdx;
  int trainIdx;
  int imgIdx;
  
  float distance;
};

extern "C" {

  CvSeq *DMatchToCvSeq( vector<DMatch> &matches, CvMemStorage *storage )
  {
    CvSeq *seq = cvCreateSeq( 0, sizeof( CvSeq ), sizeof( CvDMatch_t ), storage );

    CvSeqWriter writer;
    cvStartAppendToSeq( seq, &writer );
    for( vector<DMatch>::iterator itr = matches.begin(); itr != matches.end(); itr++ ) {

      CvDMatch_t dm;
      dm.queryIdx = (*itr).queryIdx;
      dm.trainIdx = (*itr).trainIdx;
      dm.imgIdx   = (*itr).imgIdx;
      dm.distance = (*itr).distance;
      CV_WRITE_SEQ_ELEM( dm, writer );
    }
    cvEndWriteSeq( &writer );

    printf("After conversion, vector size = %d, CvSeq size = %d\n", 
        matches.size(), seq->total );

    assert( matches.size() == seq->total );

    return seq;
  }

  CvSeq *bruteForceMatcher( CvMat *query, CvMat *train, CvMemStorage *storage, int normType, bool crossCheck CV_DEFAULT(false) ) 
  {
  
    BruteForceMatcher< L2<float> > matcher;  //( normType, crossCheck );
    Mat _train( train );
    Mat _query( query );
    vector<DMatch> matches;

    matcher.match( query, train, matches );


    return DMatchToCvSeq( matches, storage );
  }
    

}
