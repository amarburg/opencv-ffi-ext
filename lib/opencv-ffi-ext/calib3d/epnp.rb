
require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/matrix'


module CVFFI

  module Calib3d
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

#void cvSolvePnPRansac( CvMat *objectPoints, CvMat *imagePoints,
#      CvMat *cameraMatrix, Cvmat *distCoeffs,
#      CvMat *rvec, CvMat *tvec, 
#      bool useExtrinsicGuess CV_DEFAULT(false),
#      int iterationsCount CV_DEFAULT(100),
#      float reprojectionError CV_DEFAULT(8.0),
#      int minInliersCount CV_DEFAULT(100),
#      CvMat *inliers CV_DEFAULT(NULL),
#      int flags CV_DEFAULT(ITERATIVE) )

    def self.estimatePnP( objPoints, imagePoints, camera, params )

      rvec = Mat.new( 3,3, :CV_32F )
      tvec = Mat.new( 3,1, :CV_32F )
      inliers = Mat.new(1,1,:CV_32F )
      cvSolvePnPRansac( objPoints, imagePoints,
                        camera.to_CvMat, nil, rvec, tvec,
                        params.use_extrinsic_guess,
                        params.iterations_count,
                        params.reprojection_error,
                        params.min_inliers_count,
                        inliers,
                        CvPnPMethod[params.flags] )

      [rvec, tvec, inliers]
    end
  end
end
  
