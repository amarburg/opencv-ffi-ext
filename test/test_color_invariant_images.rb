
require 'test/setup'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext/color_invariance'
require 'opencv-ffi-ext/features2d/harris_with_response'

class TestColorInvariantImages < Test::Unit::TestCase

  include CVFFI::ColorInvariance

  def setup
    @img_one = TestSetup::grafitti_image
    @harris_params = CVFFI::GoodFeaturesParams.new( use_harris: true, quality_level: 0.1,
                                          k: 0.04 )
  end

  def test_spatial_quasi_invariant_image
    img = CVFFI::cvCreateMat( @img_one.height, @img_one.width, :CV_8UC1 )
    cvSpatialQuasiInvariantImage( :H_QUASI_INVARIANT, @img_one, img )

m = CVFFI::Mat.new( img )
p m
puts m.minMaxLoc


    TestSetup::save_image("h_invariant", img )
  end


end
