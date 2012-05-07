
require 'nice-ffi'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/misc/params'

require 'opencv-ffi-ext/features2d/keypoint'

module CVFFI

  module Features2D
    module HarrisCommon
      extend NiceFFI::Library

      libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
      pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
      load_library("cvffi", pathset)


      class HarrisLaplaceParams < NiceFFI::Struct
        layout :octaves, :int,
          :corn_thresh, :float,
          :dog_thresh, :float,
          :max_corners, :int,
          :num_layers, :int
      end

      class Params < CVFFI::Params
        param :octaves, 6
        param :corn_thresh, 0.01
        param :dog_thresh, 0.01
        param :max_corners, 5000
        param :num_layers, 4

        def to_HarrisLaplaceParams
          HarrisLaplaceParams.new( @params  )
        end
      end

      def self.detect( image, params = HarrisCommon::Params.new )
        params = params.to_HarrisLaplaceParams unless params.is_a?( HarrisLaplaceParams )

        kp_ptr = FFI::MemoryPointer.new :pointer
        storage = CVFFI::cvCreateMemStorage( 0 )

        image = image.ensure_greyscale

        seq_ptr = actual_detector( image, storage, params )

        keypoints = CVFFI::CvSeq.new( seq_ptr )
        #puts "Returned #{keypoints.total} keypoints"

        wrap_output( keypoints, storage )
      end
 
    end

    module HarrisLaplace
      include HarrisCommon

      attach_function :cvHarrisLaplaceDetector, [:pointer, :pointer, HarrisLaplaceParams.by_value ], CvSeq.typed_pointer

      def actual_detector( *args ); cvHarrisLaplaceDetector( *args ); end
      def wrap_output( args ); Keypoints.new( *args ); end
   end

    module HarrisAffine
      include HarrisCommon

      attach_function :cvHarrisAffineDetector, [:pointer, :pointer, HarrisLaplaceParams.by_value ], CvSeq.typed_pointer
      def actual_detector( *args ); cvHarrisAffineDetector( *args ); end
      def wrap_output( args ); EllipticKeypoints.new( *args ); end

    end
  end
end
