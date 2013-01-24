
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
                               [-1.0, -1.0, 0.0] ] ).to_CvMat
      @images = Matrix.rows( [ [0.0, 0.0],
                           [1.0, 1.0],
                           [-1.0, 1.0],
                           [1.0, -1.0],
                           [-1.0, -1.0] ] ).to_CvMat
      @camera = Mat.eye(3)
      @params = Calib3d::PnPParams.new
    end


    def test_pnp
      r,t = Calib3d::solvePnP( @objects, @images, @camera, @params )

      r.print "r"
      t.print "t"
    end

    def test_pnp_ransac
      r,t,inliers = Calib3d::estimatePnP( @objects, @images, @camera, @params )

      r.print "r"
      t.print "t"
      inliers.print "inliers"
    end

    def test_pnp_gross_outlier
      images = Matrix.rows( [ [0.0, 0.0],
                           [1.0, 1.0],
                           [-1.0, 1.0],
                           [1.0, -1.0],
                           [100.0, -1.0] ] ).to_CvMat

      r,t,inliers = Calib3d::estimatePnP( @objects, images, @camera, @params )

      r.print "r"
      t.print "t"
      inliers.print "inliers"
    end
end

