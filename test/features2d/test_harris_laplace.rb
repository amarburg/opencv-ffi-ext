
require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/harris_laplace'
require 'opencv-ffi-ext/features2d/keypoint'

require 'opencv-ffi-wrappers/misc/each_two'


class TestHarrisLaplace < Test::Unit::TestCase
  include CVFFI::Features2D

  def setup
    @img = TestSetup::tiny_test_image
    @kp_ptr = FFI::MemoryPointer.new :pointer
    @mem_storage = CVFFI::cvCreateMemStorage( 0 )
  end
  
  def test_HarrisLaplace
    detector_test_common( HarrisLaplace, Keypoints )
  end

  def test_HarrisAffine
    detector_test_common( HarrisAffine, EllipticKeypoints )
  end

  def detector_test_common( detectorKlass, keypointKlass )
    params = HarrisCommon::Params.new

    params.max_corners = 2
    kps = detectorKlass::detect( @img, params )
    assert kps.length <= 2

    # TODO:  Don't believe these detectors will take '0' corners yet.
    params.max_corners = 1000000
    kps = detectorKlass::detect( @img, params )

    assert_not_nil kps

    puts "The #{detectorKlass} detector found #{kps.size} keypoints"
    puts "First keypoint: #{kps.first.inspect}"

    ## Test serialization and unserialization
    asYaml = kps.to_yaml
    unserialized = keypointKlass.from_a( asYaml )

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
