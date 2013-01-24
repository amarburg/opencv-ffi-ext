
require 'test/setup'
require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext'

class TestPnP < Test::Unit::TestCase
    include CVFFI

    def test_pnp
      objects = Matrix.rows( [ [0.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0],
                               [-1.0, 1.0, 0.0],
                               [1.0, -1.0, 0.0],
                               [-1.0, -1.0, 0.0] ] ).to_CvMat
      images = Matrix.rows( [ [0.0, 0.0],
                           [1.0, 1.0],
                           [-1.0, 1.0],
                           [1.0, -1.0],
                           [-1.0, -1.0] ] ).to_CvMat
      camera = Mat.eye(3)
      params = Calib3d::PnPParams.new

      r,t,inliers = Calib3d::estimatePnP( objects, images, camera, params )

      r.print "r"
      t.print "t"
      inliers.print "inliers"

    end
end

