
require 'opencv-ffi-wrappers'


module CVFFI

  module Calib3d
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    class FEstimatorParams < CVFFI::Params
      param :max_iters, 2000
      param :outlier_threshold, 3
      param :confidence, 0.99
      param :method, :CV_FM_RANSAC
    end

    attach_function :cvEstimateFundamental, [ :pointer, :pointer, :pointer, 
      :int, :double, :double, :int, :pointer ], :int

    def self.estimateFundamental( points1, points2, params )

      fundamental = CVFFI::cvCreateMat( 3,3, :CV_32F )
      status = CVFFI::cvCreateMat( points1.height, 1, :CV_8U )

      puts "Running my fundamental calculation."
      #ret = CVFFI::cvFindFundamentalMat( points1, points2, fundamental, method, param1, param2, status )
      ret = cvEstimateFundamental( points1, points2, fundamental, CvRansacMethod[ params.method ], params.outlier_threshold, params.confidence, params.max_iters, status )

      if ret> 0
        Fundamental.new( fundamental, status, ret )
      else
        nil
      end

    end
  end
end
