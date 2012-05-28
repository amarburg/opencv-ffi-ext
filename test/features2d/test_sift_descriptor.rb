
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/sift'

class TestSIFT < Test::Unit::TestCase
  include CVFFI::Features2D

  def setup
    @img = TestSetup::small_test_image
  end


  def test_SIFTDescribe
  keypoints = [ [100,100], [50,50] ]
  keypoints = keypoints.map { |kp|
	  (SIFT::CvSIFTFeature.num_keys-1-kp.length).times {
	  kp.push 0
	  }
	  kp.push ""

kp[6] = kp[0]
kp[7] = kp[1]
# If the feature is forced to be at octave 0, interval 0, the scl and feature_data.scl_octv should be SIGMA = 1.6
kp[2] = kp[11] = 1.6
		  kp
 }
p keypoints

kps = SIFT::Results.from_a( keypoints )
kps.each { |kp| p kp; p kp.feature_data }

params = SIFT::Params.new
params.recalculateAngles = 1;
SIFT::detect_describe( @img, params, kps )

kps.each { |kp|
p kp.to_a.join(',')
}
	end


	end
