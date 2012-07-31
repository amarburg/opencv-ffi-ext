
require 'test/setup'

require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext/color_invariance'

class TestChuColorInvariance < Test::Unit::TestCase

  include CVFFI::ColorInvariance

  def setup
    @img_one = TestSetup::test_image
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

  def test_chu_harris
    img = @img_one.clone
    assert_not_nil img

    params = CVFFI::GoodFeaturesParams.new( use_harris: true, quality_level: 0.5,
                                          k: 0.04 )
    corners = chuQuasiInvariantFeatures( img, params )
    puts "Chu quasi invariant found #{corners.length} features"

    draw_and_save_keypoints( img, corners, "chu_quasiinvariant_harris" )
  end


  def test_chu_harris
    img = @img_one.clone
    assert_not_nil img

    params = CVFFI::GoodFeaturesParams.new( use_harris: true, quality_level: 0.5,
                                          k: 0.04 )
    corners = chuQuasiInvariantFeatures( img, params )
    puts "Chu quasi invariant found #{corners.length} features"

    feature_img = img.clone
    corners.each { |corner|
      puts "Corner at #{corner.x} x #{corner.y}"
      CVFFI::cvCircle( feature_img, CVFFI::CvPoint.new( :x => corner.x, :y => corner.y ), 20,
                                            CVFFI::CvScalar.new( :w=>255, :x=>255, :y=>0, :z=>0 ), -1, 8, 0 )
    }
    TestSetup::save_image("chu_invariant_features", feature_img )

  end



end
