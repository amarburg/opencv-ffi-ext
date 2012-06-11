#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core_c.h>
#include <vector>

#include <stdio.h>

using namespace cv;

// The enumeration of the different types of Norms comes from OpenCV 2.4.x
// Backported here for convience.
//enum { NORM_INF=1, NORM_L1=2, NORM_L2=4, NORM_L2SQR=5, NORM_HAMMING=6, NORM_HAMMING2=7, NORM_TYPE_MASK=7, NORM_RELATIVE=8, NORM_MINMAX=32 };
enum { NORM_L2SQR = 5, NORM_HAMMING=6 };

struct CvDMatch_t {
  int queryIdx;
  int trainIdx;
  int imgIdx;

  float distance;
};

enum { CONVERT_ALL = 0, TAKE_JUST_FIRST = 1 };

extern "C" {

  static void writeDmatchToSeqWriter( CvSeqWriter &writer, const DMatch &dmatch )
  {
    CvDMatch_t dm;
    dm.queryIdx = dmatch.queryIdx;
    dm.trainIdx = dmatch.trainIdx;
    dm.imgIdx   = dmatch.imgIdx;
    dm.distance = dmatch.distance;
    CV_WRITE_SEQ_ELEM( dm, writer );

  }

  CvSeq *DMatchToCvSeq( const vector< vector<DMatch> > &matches, CvMemStorage *storage, int doTakeFirst )
  {
    // TODO:  For now, Knn will return a flattened set of matches, may do this differently in the future
    CvSeq *seq = cvCreateSeq( 0, sizeof( CvSeq ), sizeof( CvDMatch_t ), storage );

    CvSeqWriter writer;
    cvStartAppendToSeq( seq, &writer );
    for( vector< vector<DMatch> >::const_iterator itr = matches.begin(); itr != matches.end(); itr++ ) {

      if( doTakeFirst == TAKE_JUST_FIRST ) {
        if( !(*itr).empty() ) {
          writeDmatchToSeqWriter( writer, (*itr)[0] );
        }
      } else {
        // Take then all
        for( vector<DMatch>::const_iterator itr2 = (*itr).begin();  itr2 != (*itr).end(); itr2++ ) {
          writeDmatchToSeqWriter( writer, (*itr2) );
        }
      }
    }
    cvEndWriteSeq( &writer );

    //printf("After conversion, vector size = %d, CvSeq size = %d\n", 
    //    matches.size(), seq->total );

    //assert( matches.size() == (unsigned int)seq->total );

    return seq;
  }


  //##### Brute Force Matcher #######
  void bruteForceMatcherKnnActual( CvMat *query, CvMat *train, vector< vector<DMatch> > &matches, int normType, int knn, bool crossCheck CV_DEFAULT(false) ) 
  {
    Mat _train( train );
    Mat _query( query );

    if( normType == NORM_L2 ) {
      BruteForceMatcher< L2<float> > matcher;  //( normType, crossCheck );
      matcher.knnMatch( query, train, matches, knn );
    } else if (normType == NORM_L2SQR ) {
      BruteForceMatcher< SL2<float> > matcher;  //( normType, crossCheck );
      matcher.knnMatch( query, train, matches, knn );
    } else if (normType == NORM_L1 ) {
      BruteForceMatcher< L1<float> > matcher;  //( normType, crossCheck );
      matcher.knnMatch( query, train, matches, knn );
    } else if (normType == NORM_HAMMING ) {
      BruteForceMatcher< Hamming > matcher;  //( normType, crossCheck );
      matcher.knnMatch( query, train, matches, knn );
    } else {
      printf("bruteForceMatcherKnn doesn't understand norm type %d\n", normType);
    }
  }

  // bruteForceMatcherKnn and bruteForceMatcher require different strategies for 
  // convtering the vector<vector<DMatch>> to a CvSeq
  CvSeq *bruteForceMatcherKnn( CvMat *query, CvMat *train, CvMemStorage *storage, int normType, int knn, bool crossCheck CV_DEFAULT(false) ) 
  {
    vector< vector<DMatch> > matches;
    bruteForceMatcherKnnActual( query, train, matches, normType, knn, crossCheck );
    return DMatchToCvSeq( matches, storage, CONVERT_ALL );
  }

  CvSeq *bruteForceMatcher( CvMat *query, CvMat *train, CvMemStorage *storage, int normType, bool crossCheck CV_DEFAULT(false) ) 
  {
    vector< vector<DMatch> > matches;
    bruteForceMatcherKnnActual( query, train, matches, normType, 1, crossCheck );
    return DMatchToCvSeq( matches, storage, TAKE_JUST_FIRST );
  }

  CvSeq *bruteForceMatcherRadius( CvMat *query, CvMat *train, CvMemStorage *storage, int normType, float maxDistance, bool crossCheck CV_DEFAULT(false) ) 
  {
    Mat _train( train );
    Mat _query( query );
    vector< vector<DMatch> > matches;

    if( normType == NORM_L2 ) {
        BruteForceMatcher< L2<float> > matcher;  //( normType, crossCheck );
        matcher.radiusMatch( query, train, matches, maxDistance );
    } else if (normType == NORM_L2SQR ) {
        BruteForceMatcher< SL2<float> > matcher;  //( normType, crossCheck );
        matcher.radiusMatch( query, train, matches, maxDistance );
    } else if (normType == NORM_L1 ) {
        BruteForceMatcher< L1<float> > matcher;  //( normType, crossCheck );
        matcher.radiusMatch( query, train, matches, maxDistance );
    } else if (normType == NORM_HAMMING) {
        BruteForceMatcher< Hamming > matcher;  //( normType, crossCheck );
        matcher.radiusMatch( query, train, matches, maxDistance );
    } else {
        printf("bruteForceMatcherRadius doesn't understand norm type %d\n", normType);
    }

    return DMatchToCvSeq( matches, storage, CONVERT_ALL );
  }


  //##### FLANN Matcher #######
  // TODO:  Expose flann parameters through API
  void flannMatcherKnnActual( CvMat *query, CvMat *train, vector< vector<DMatch> > &matches, int knn )
  {
    Mat _train( train );
    Mat _query( query );

    FlannBasedMatcher matcher;
    matcher.knnMatch( query, train, matches, knn );
  }

  CvSeq *flannBasedMatcherKnn( CvMat *query, CvMat *train, CvMemStorage *storage, int knn )
  {
    vector< vector<DMatch> > matches;
    flannMatcherKnnActual( query, train, matches, knn );
    return DMatchToCvSeq( matches, storage, CONVERT_ALL );
  }

  CvSeq *flannBasedMatcher( CvMat *query, CvMat *train, CvMemStorage *storage )
  {
    vector< vector<DMatch> > matches;
    flannMatcherKnnActual( query, train, matches,  1 );
    return DMatchToCvSeq( matches, storage, TAKE_JUST_FIRST );
  }


  CvSeq *flannBasedMatcherRadius( CvMat *query, CvMat *train, CvMemStorage *storage, float maxDistance )
  {
    Mat _train( train );
    Mat _query( query );
    vector< vector<DMatch> > matches;

    FlannBasedMatcher matcher;
    matcher.radiusMatch( query, train, matches, maxDistance );
    return DMatchToCvSeq( matches, storage, CONVERT_ALL );
  }



}
