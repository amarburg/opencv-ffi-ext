
require 'test/setup'
require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext'

class TestPnP < Test::Unit::TestCase
  include CVFFI

  def setup
    @objects = Matrix.rows( [ [0.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [-1.0, 1.0, 0.0],
                           [1.0, -1.0, 0.0],
                           [2.0, 2.0, 0.0],
                           [-1.0, -1.0, 0.0] ] )
    @images = Matrix.rows( [ [0.0, 0.0],
                          [1.0, 1.0],
                          [-1.0, 1.0],
                          [1.0, -1.0],
                          [2.0, 2.0],
                          [-1.0, -1.0] ] )
    @camera = Mat.eye(3)
    @params = Calib3d::PnPParams.new
  end


  def test_pnp
    r,t = Calib3d::solvePnP( @objects, @images, @camera, @params )

    3.times { |i| assert_in_delta 0.0, r.at(i,0), 1e-6 }
    2.times { |i| assert_in_delta 0.0, t.at(i,0), 1e-6 }
    assert_in_delta 1.0, t.at(2,0), 1e-6
  end

  def test_pnp_ransac
    r,t,inliers = Calib3d::solvePnPRansac( @objects, @images, @camera, @params )

    3.times { |i| assert_in_delta 0.0, r.at(i,0), 1e-6 }
    2.times { |i| assert_in_delta 0.0, t.at(i,0), 1e-6 }
    assert_in_delta 1.0, t.at(2,0), 1e-6
    @objects.row_size.times { |i|
      assert_equal 1.0, inliers.at(i,0)
    }
  end

  def test_pnp_gross_outlier
    images = Matrix.rows( [ [0.0, 0.0],
                         [1.0, 1.0],
                         [-1.0, 1.0],
                         [1.0, -1.0],
                         [2.0, 2.0],
                         [100.0, -1.0] ] ).to_CvMat
    # With this sample data, I need to dial down the reproj error
    # to get RANSAC to kick the bad point out...
    params = @params.clone
    params.reprojection_error = 1.0

    r,t,inliers = Calib3d::solvePnPRansac( @objects, images, @camera, @params )

    3.times { |i| assert_in_delta 0.0, r.at(i,0), 1e-6 }
    2.times { |i| assert_in_delta 0.0, t.at(i,0), 1e-6 }
    assert_in_delta 1.0, t.at(2,0), 1e-6
    (@objects.row_size-1).times { |i|
      assert_equal 1.0, inliers.at(i,0)
    }
    assert_equal 0.0, inliers.at( (@objects.row_size-1), 0 )
  end
end

