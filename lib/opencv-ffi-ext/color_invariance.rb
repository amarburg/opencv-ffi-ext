
require 'opencv-ffi-wrappers/imgproc/features'

module CVFFI
  module ColorInvariance
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

    attach_function :cvGenerateQuasiInvariant, [ :int, :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateSQuasiInvariant, [ :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateHQuasiInvariant, [ :pointer, :pointer, :pointer ], :void
    attach_function :cvGenerateChuQuasiInvariants, [ :pointer, :pointer ], :void

    attach_function :cvQuasiInvariantFeaturesToTrack, [ :int, :pointer, :pointer, 
                                                   :pointer, :double, :double, :int,
                                                   :pointer, :int, :double ], :void


  def quasiInvariantFeatures( which, m, params = CVFFI::GoodFeaturesParams.new )

    max_corners = FFI::MemoryPointer.new :int
    max_corners.write_int params.max_corners
    corners = FFI::MemoryPointer.new( CVFFI::CvPoint2D32f, params.max_corners )

    cvQuasiInvariantFeaturesToTrack( which, m.to_CvMat, 
                                    corners, max_corners, 
                                 params.quality_level, params.min_distance, params.block_size, params.mask,
                                 params.use_harris ? 1 : 0, params.k )

    num_corners = max_corners.read_int 
    points = Array.new( num_corners ) {|i|
      CVFFI::CvPoint2D32f.new( corners + CVFFI::CvPoint2D32f.size * i )
    }
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

