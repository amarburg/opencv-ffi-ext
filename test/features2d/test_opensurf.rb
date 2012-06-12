

require 'test/setup'

require 'opencv-ffi'
require 'opencv-ffi-ext/features2d/opensurf'

class TestOpenSURF < Test::Unit::TestCase
  include CVFFI

  def setup
  end


  def test_openSurfDetect
    img = TestSetup::test_image
    params = OpenSURF::Params.new

    # This should test the auto=conversion to greyscale
    surf = OpenSURF::detect( img, params )

    assert_not_nil surf

    surf.mark_on_image( img, {:radius=>5, :thickness=>-1} )
    CVFFI::cvSaveImage( TestSetup::output_filename("openSurfPts.jpg"), img )

    puts "OpenSURF detected #{surf.length} points"


    descriptors = OpenSURF::describe( img, surf, params )

    puts "After description #{descriptors.length} points"
 end

  def test_openSurf_serialization
    img = TestSetup::test_image
    params = OpenSURF::Params.new

    # This should test the auto=conversion to greyscale
    surf = OpenSURF::detect( img, params )

    assert_not_nil surf

    as_a = surf.to_a
    unserialized = OpenSURF::Results.from_a as_a

    assert_equal surf.length, unserialized.length

    surf.extend EachTwo
    surf.each2( unserialized ) { |s,u|
      assert_equal s,u
    }

  end

end
