
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/harris_with_response'

require 'opencv-ffi-wrappers/misc/each_two'


class TestHarrisWithResponse < Test::Unit::TestCase
  include CVFFI::Features2D

  def setup
    @img = TestSetup::small_test_image
  end
  
  def test_HarrisWithResponse
    params = CVFFI::GoodFeaturesParams.new
    params.use_harris = true
    detector_test_common( params, "Harris" )
  end

  def test_ShiTomasiWithResponse
    params = CVFFI::GoodFeaturesParams.new
    params.use_harris = false
    detector_test_common( params, "Shi-Tomasi" )
  end

  def detector_test_common( params, name )
    params.max_corners = 2
    kps = HarrisWithResponse::detect( @img, params )
    assert kps.length <= 2

    params.max_corners = 0
    p params
    kps = HarrisWithResponse::detect( @img, params )
    assert_not_nil kps

    puts "The #{name} (with response) detector found #{kps.size} keypoints"
    puts "First keypoint: " + kps.first.inspect

    ## Test serialization and unserialization
    asYaml = kps.to_yaml
    unserialized = Keypoints.from_a( asYaml )

    assert_equal kps.length, unserialized.length

    kps.extend EachTwo
    kps.each_two( unserialized ) { |kp,uns|
      assert_equal kp, uns

      assert kp.x > 0.0 and kp.y > 0.0
      assert kp.y < @img.height, "Image height"
      assert kp.x < @img.width, "Image width"

    }
  end


end
