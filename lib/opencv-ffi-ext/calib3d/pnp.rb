
require 'opencv-ffi'
require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/matrix'


module CVFFI

  module Calib3d
    include CVFFI
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    CvPnPMethod = enum :cvPnPMethod, [ :CV_ITERATIVE, 0,
                                       :CV_EPNP, 1,
                                       :CV_P3P, 2 ]

    class PnPParams < CVFFI::Params
      param :iterations_count, 100
      param :reprojection_error, 8.0
      param :min_inliers_count, 100
      param :use_extrinsic_guess, false
      param :flags, :CV_ITERATIVE
    end

    attach_function :cvSolvePnPRansac, [ :pointer, :pointer, :pointer, :pointer, :pointer, :pointer, :bool, :int, :float, :int, :pointer, :cvPnPMethod ], :void

    attach_function :cvSolvePnP, [ :pointer, :pointer, :pointer, :pointer, :pointer, :pointer, :bool, :cvPnPMethod ], :void

#void cvSolvePnPRansac( CvMat *objectPoints, CvMat *imagePoints,
#      CvMat *cameraMatrix, Cvmat *distCoeffs,
#      CvMat *rvec, CvMat *tvec, 
#      bool useExtrinsicGuess CV_DEFAULT(false),
#      int iterationsCount CV_DEFAULT(100),
#      float reprojectionError CV_DEFAULT(8.0),
#      int minInliersCount CV_DEFAULT(100),
#      CvMat *inliers CV_DEFAULT(NULL),
#      int flags CV_DEFAULT(ITERATIVE) )


# Calls the non-ransac pnp algorithm
   def self.solvePnP( objPoints, imagePoints, camera, params )

      rvec = CVFFI::cvCreateMat( 3, 1, :CV_64F )
      tvec = CVFFI::cvCreateMat( 3, 1, :CV_64F )

      cvSolvePnP( objPoints.to_CvMat, imagePoints.to_CvMat,
                        camera.to_CvMat, nil, rvec, tvec,
                        params.use_extrinsic_guess,
                        CvPnPMethod[params.flags] )

      [rvec.to_Mat, tvec.to_Mat]
    end
 

   def self.solvePnPRansac( objPoints, imagePoints, camera, params )

      # As solvePnPRansac modifies rvec and tvec it's fairly important
      # that these pointers match up for copying.  
      #
      # Opencv's function defines "inliers" as an array of inlier
      # indices .. so it's variable length depending on how
      # many inliers there are.  For the C interface I've redefined
      # it as a status mask, as per the cvFundamental* functions.
      # So it's fixed length
      rvec = CVFFI::cvCreateMat( 3, 1, :CV_64F )
      tvec = CVFFI::cvCreateMat( 3, 1, :CV_64F )
      inliers = CVFFI::cvCreateMat( objPoints.height, 1, :CV_8U )

      cvSolvePnPRansac( objPoints.to_CvMat, imagePoints.to_CvMat,
                        camera.to_CvMat, nil, rvec, tvec,
                        params.use_extrinsic_guess,
                        params.iterations_count,
                        params.reprojection_error,
                        params.min_inliers_count,
                        inliers,
                        CvPnPMethod[params.flags] )

      [rvec.to_Mat, tvec.to_Mat, inliers.to_Mat]
    end

  end
end
  
