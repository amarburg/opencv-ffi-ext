
require 'opencv-ffi-wrappers/imgproc/features'
require 'opencv-ffi-ext/features2d/harris_with_response'
require 'opencv-ffi-wrappers/core/sequence'
require 'opencv-ffi/core/types'

module CVFFI
  module ColorInvariance
    include CVFFI::Features2D::HarrisWithResponse

    extend NiceFFI::Library
    libs_dir = File.dirname(__FILE__) + "/../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )

    load_library("cvffi", pathset)

    enum :cvColorInvariance, [ :CV_COLOR_INVARIANCE_PASSTHROUGH, 0,
      :CV_COLOR_INVARIANCE_RGB2GAUSSIAN_OPPONENT, 1,
      :CV_COLOR_INVARIANCE_BGR2GAUSSIAN_OPPONENT, 2,
      :CV_COLOR_INVARIANCE_Gray2YB, 100,
      :CV_COLOR_INVARIANCE_Gray2RG, 101]

    #  void cvCvtColorInvariants( const CvArr *srcarr, CvArr *dstarr, int code )
    attach_function :cvCvtColorInvariants, [ :pointer, :pointer, :int ], :void


    attach_function :cvNormalizedColorImage, [ :pointer, :pointer ], :void
    attach_function :cvGenerateColorTensor, [ :pointer, :pointer, :pointer ], :void

    enum :quasiInvariants, [ :H_QUASI_INVARIANT, 0,
      :S_QUASI_INVARIANT, 
      :HS_QUASI_INVARIANTS, 
      :GREYSCALE ]

attach_function :cvSpatialQuasiInvariantImage, [ :int, :pointer, :pointer ], :void
    attach_function :cvGenerateQuasiInvariant, [ :int, :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateSQuasiInvariant, [ :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateHQuasiInvariant, [ :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateChuQuasiInvariants, [ :pointer, :pointer ], :void

    attach_function :cvFoobar, [:int, :pointer, :pointer, :pointer, HarrisParams.by_value ], :void
    attach_function :cvQuasiInvariantFeatures, [:int, :pointer, :pointer, :pointer, HarrisParams.by_value ], :pointer #CVFFI::CvSeq.typed_pointer

    #    attach_function :cvQuasiInvariantFeaturesToTrack, [ :int, :pointer, :pointer, 
     #                                                  :pointer, :double, :double, :int,
     #                                                  :pointer, :int, :double ], :void
    #
    #
    #  def quasiInvariantFeatures( which, m, params = CVFFI::GoodFeaturesParams.new )
    #
    #    max_corners = FFI::MemoryPointer.new :int
    #    max_corners.write_int params.max_corners
    #    corners = FFI::MemoryPointer.new( CVFFI::CvPoint2D32f, params.max_corners )
    #
    #    cvQuasiInvariantFeaturesToTrack( which, m.to_CvMat, 
    #                                    corners, max_corners, 
    #                                 params.quality_level, params.min_distance, params.block_size, params.mask,
    #                                 params.use_harris ? 1 : 0, params.k )
    #
    #    num_corners = max_corners.read_int 
    #    points = Array.new( num_corners ) {|i|
    #      CVFFI::CvPoint2D32f.new( corners + CVFFI::CvPoint2D32f.size * i )
    #    }
    #  end



    def quasiInvariantFeatures( which, image, params = CVFFI::GoodFeaturesParams.new )

      params = params.to_HarrisParams unless params.is_a?( HarrisParams )
      storage = CVFFI::cvCreateMemStorage( 0 )

      seq_ptr = cvQuasiInvariantFeatures( which, image, nil, storage, params )

      keypoints = CVFFI::CvSeq.new( seq_ptr )
      CVFFI::Features2D::Keypoints.new( keypoints, storage )
    end

    def chuQuasiInvariantFeatures( m, params = CVFFI::GoodFeaturesParams.new )
      quasiInvariantFeatures( :HS_QUASI_INVARIANTS, m, params )
    end

    def hQuasiInvariantFeatures( m, params = CVFFI::GoodFeaturesParams.new )
      quasiInvariantFeatures( :H_QUASI_INVARIANT, m, params )
    end

    def greyscaleQuasiInvariantFeatures( m, params = CVFFI::GoodFeaturesParams.new )
      quasiInvariantFeatures( :GREYSCALE, m, params )
    end

  end
end

