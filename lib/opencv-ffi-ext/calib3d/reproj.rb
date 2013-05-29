
require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/matrix'


module CVFFI

  module Calib3d
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    attach_function :cvFMaxReprojError, [ CvMat.by_ref, CvMat.by_ref, CvMat.by_ref, CvMat.by_ref ], :void 

    def self.computeReprojError( x, xp, f )
      # The C function assumes doubles for x, xp, and f, but returns floats
      x  =  x.to_Mat( :type => :CV_64F ).to_CvMat
      xp = xp.to_Mat( :type => :CV_64F ).to_CvMat
      f  =  f.to_Mat( :type => :CV_64F ).to_CvMat
      err = CVFFI::cvCreateMat( x.height, 1, :CV_32F )

      cvFMaxReprojError( x, xp, f, err )

      Mat.new(err )
    end

    attach_function :cvHMaxReprojError, [ CvMat.by_ref, CvMat.by_ref, CvMat.by_ref, CvMat.by_ref ], :void 

    def self.computeHomographyReprojError( x, xp, f )
      # The C function assumes doubles for x, xp, and f, but returns floats
      x  =  x.to_Mat( :type => :CV_64F ).to_CvMat
      xp = xp.to_Mat( :type => :CV_64F ).to_CvMat
      f  =  f.to_Mat( :type => :CV_64F ).to_CvMat
      err = CVFFI::cvCreateMat( x.height, 1, :CV_32F )

      cvHMaxReprojError( x, xp, f, err )

      Mat.new(err )
    end

  end
end
