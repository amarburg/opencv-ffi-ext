
require 'nice-ffi'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-wrappers/misc/params'

require 'opencv-ffi-ext/features2d/keypoint'


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
          :k, :double,
          :use_harris, :int,
          :max_corners, :int
      end

      class CVFFI::GoodFeaturesParams

        def to_HarrisParams
          # HarrisParams doesn't have a mask field
          params = @params
          params.delete( :mask )
          params[:use_harris] = case params[:use_harris] 
                                when false,0
                                  0
                                when true,1
                                  1
                                end
          HarrisParams.new( params )
        end
      end

      # CvSeq *cvGoodFeaturesWithResponse( const void* _image,
      #                                    const void* _maskImage,
      #                                    CvMemStorage *pool, 
      #                                    const NewHarrisParams_t para
      attach_function :cvGoodFeaturesWithResponse, [:pointer, :pointer, :pointer, HarrisParams.by_value ], CvSeq.typed_pointer

      def self.detect( image, params = CVFFI::GoodFeaturesParams.new )
        params = params.to_HarrisParams unless params.is_a?( HarrisParams )
        storage = CVFFI::cvCreateMemStorage( 0 )

        seq_ptr = cvGoodFeaturesWithResponse( image.ensure_greyscale, nil, storage, params )

        keypoints = CVFFI::CvSeq.new( seq_ptr )
        CVFFI::Features2D::Keypoints.new( keypoints, storage )
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
