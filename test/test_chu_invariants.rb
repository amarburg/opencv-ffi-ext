
require 'test/setup'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext/color_invariance'
require 'opencv-ffi-ext/features2d/harris_with_response'

class TestChuColorInvariance < Test::Unit::TestCase

  include CVFFI::ColorInvariance

  def setup
    @img_one = TestSetup::test_image
    @harris_params = CVFFI::GoodFeaturesParams.new( use_harris: true, quality_level: 0.1,
                                          k: 0.04 )
  end

  def test_chu_color_invariants
    img = @img_one.clone
    assert_not_nil img

    chu_invariants = CVFFI::cvCreateMat( img.height, img.width, :CV_8UC4 )
    #chu_invariants = CVFFI::cvCreateMat( img.height, img.width, :CV_32FC4 )

    cvGenerateChuQuasiInvariants( img, chu_invariants )

    ## Break out the four channels...
    
    hx,hy,sx,sy = chu_invariants.split
    TestSetup::save_image("chu_invariants_hx", hx )
    TestSetup::save_image("chu_invariants_hy", hy )
    TestSetup::save_image("chu_invariants_sx", sx )
    TestSetup::save_image("chu_invariants_sy", sy )
  end

  def test_spatial_quasi_invariant_image
    img = CVFFI::cvCreateMat( 1,1, :CV_32F )
    cvSpatialQuasiInvariantImage( :H_QUASI_INVARIANT, @img_one, img )
    TestSetup::save_image("h_invariant", img )
  end

  def test_chu_harris
    img = @img_one.clone
    assert_not_nil img

    puts "*** Computing Chu quasi invariants."
    corners = chuQuasiInvariantFeatures( img, @harris_params )
    puts "Chu quasi invariant found #{corners.length} features"

    TestSetup::draw_and_save_keypoints( img, corners, "chu_invariant_harris" )
  end


  def test_h_harris
    img = @img_one.clone
    assert_not_nil img

    puts "*** Computing H quasi invariants."
    corners = hQuasiInvariantFeatures( img, @harris_params )
    puts "H quasi invariant found #{corners.length} features"
    TestSetup::draw_and_save_keypoints( img, corners, "h_invariant_harris" )
  end

  def test_grey_harris
    img = @img_one.clone
    assert_not_nil img

    puts "*** Computing greyscale quasi invariants."
    corners = greyscaleQuasiInvariantFeatures( img, @harris_params )
    puts "Grey quasi invariant found #{corners.length} features"
    TestSetup::draw_and_save_keypoints( img, corners, "grey_invariant_harris" )

    puts "---- Computing harris with response."
    kps = CVFFI::Features2D::HarrisWithResponse::detect( img, @harris_params )
    puts "Harris found #{kps.length} features"
  end




end
