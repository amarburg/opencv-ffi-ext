
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/sift'

class TestSIFT < Test::Unit::TestCase
  include CVFFI
  include CVFFI::Features2D

  def setup
    @img = TestSetup::small_test_image
  end


  def test_SIFT_describe_coerced_keypoints
    keypoints = [ [100,100], [50,50] ].map { |pt|
      Point.new(pt)
    }

    keypoints = SIFT::Results.coerce( keypoints )
    params = SIFT::Params.new
    params.recalculateAngles = 1;
    kps = SIFT::detect_describe( @img, params, keypoints )

    kps.each { |kp|
      p kp
      p kp.feature_data
    }
  end


end
