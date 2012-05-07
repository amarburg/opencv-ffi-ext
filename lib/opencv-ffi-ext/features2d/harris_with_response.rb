
require 'nice-ffi'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/misc/params'


module CVFFI

  module Features2D
    module HarrisWithResponse
      extend NiceFFI::Library

      libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
      pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
      load_library("cvffi", pathset)
      
      # TODO: Move these over to -wrapper and start using it
      # instead of setting each individually..
      class HarrisParams < NiceFFI::Struct
        layout :quality_level, :double,
          :min_distance, :double,
          :block_size, :int,
          :harris_k, :double,
          :use_harris, :int,
          :max_corners, :int
      end

      class CVFFI::GoodFeaturesParams

        def to_HarrisParams
          # CvNewHarrisParams doesn't have a mask field
          p = @params
          p.delete( :mask )
          CvHarrisWithResponse.new( p )
        end
      end

      # CvSeq *cvGoodFeaturesWithResponse( const void* _image,
      #                                    const void* _maskImage,
      #                                    CvMemStorage *pool, 
      #                                    const NewHarrisParams_t para
      attach_function :cvGoodFeaturesWithResponse, [:pointer, :pointer, *pointer, HarrisParams.by_value ], CvSeq.typed_pointer

      def self.detect( image, params = CVFFI::GoodFeaturesParams.new )
        params = params.to_HarrisParams unless params.is_a?( HarrisParams )

        storage = CVFFI::cvCreateMemStorage( 0 )

        image = image.ensure_greyscale

        seq_ptr = cvHarrisLaplaceDetector( image, storage, params )

        keypoints = CVFFI::CvSeq.new( seq_ptr )
        puts "Returned #{keypoints.total} keypoints"

        Keypoints.new( keypoints, storage )
      end

      def self.harrisDetect( image, params = CVFFI::GoodFeaturesParams.new )
        params.use_harris = 1
        detect( image, params )
      end

      def self.shiTomasiDetect( image, params = CVFFI::GoodFeaturesParams.new )
        params.use_harris = 0
        detect( image, params )
      end
    end

    end
  end
end
