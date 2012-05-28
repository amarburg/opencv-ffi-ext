
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/sift'

class TestSIFT < Test::Unit::TestCase
  include CVFFI::Features2D

  def setup
    @img = TestSetup::small_test_image
  end


  def test_SIFTDescribe
  keypoints = [ [100,100], [200,200] ]
  keypoints = keypoints.map { |kp|
	  (SIFT::CvSIFTFeature.num_keys-1-kp.length).times {
	  kp.push 0
	  }
	  kp.push ""

kp[6] = kp[0]
kp[7] = kp[1]
		  kp
 }
p keypoints

kps = SIFT::Results.from_a( keypoints )
p kps

	params = SIFT::Params.new
params.recalculateAngles = 1;
kps = SIFT::detect_describe( @img, params, kps )

kps.each { |kp|
p kp.to_a.join(',')
}
	end


	end
