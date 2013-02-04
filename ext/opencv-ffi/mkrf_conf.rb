
puts ENV.keys

ENV.keys.each { |k|
  puts "#{k} : #{ENV[k]}"
}

puts ENV['GEM_PATH']

require '../mkrf-monkey'

# The compiler for availability checking must be specified as 'g++'
# otherwise it will use gcc and choke on Eigen

sources = [ "*.cpp",
            "harris_laplace/*.cpp",
            "color_invariance/*.cpp",
            "sift/*.cpp",
            "matcher/*.cpp",
            "harris/*.cpp",
            "calib3d/*.cpp" ]

Mkrf::Generator.new('libcvffi', sources, { :compiler=>"g++"}) { |g|
  g.include_library 'stdc++'
  raise "Can't find 'opencv_core'" unless g.include_library 'opencv_core', 'main', "#{ENV['HOME']}/usr/opencv-2.4/lib"

  raise "Can't find 'opencv_features2d'" unless g.include_library 'opencv_features2d', 'main', "#{ENV['HOME']}/usr/opencv-2.4/lib"

  raise "Can't find 'opencv_calib3d'" unless g.include_library 'opencv_calib3d', 'main', "#{ENV['HOME']}/usr/opencv-2.4/lib"

  raise "Can't find 'opencv_nonfree'" unless g.include_library 'opencv_nonfree', 'main', "#{ENV['HOME']}/usr/opencv-2.4/lib"
  
  #g.include_header  'eigen3/Eigen/Core', "#{ENV['HOME']}/usr/include"
  g.cflags += "-I#{ENV['HOME']}/usr/opencv-2.4/include "
}

