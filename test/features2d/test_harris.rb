
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/harris_with_response'

require 'opencv-ffi-wrappers/misc/each_two'


class TestHarrisWithResponse < Test::Unit::TestCase
  include CVFFI::Features2D

  def setup
    @img = TestSetup::tiny_test_image
  end
  
  def test_HarrisWithResponse
    params = GoodFeaturesParams.new
    params.use_harris = true
    detector_common( params, "Harris" )
  end

  def test_ShiTomasiWithResponse
    params = GoodFeaturesParams.new
    params.use_harris = false
    detector_common( params, "Shi-Tomasi" )
  end

  def detector_common( params, name )

    kps = HarrisWithReponse::detect( @img, params )

    assert_not_nil kps

    puts "The #{name} (with response) detector found #{kps.size} keypoints"

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
