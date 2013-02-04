# -*- encoding: utf-8 -*-
$:.push File.expand_path("../lib", __FILE__)
require "opencv-ffi-ext/version"

Gem::Specification.new do |s|
  s.name        = "opencv-ffi-ext"
  s.version     = CVFFI::Ext::VERSION
  s.authors     = ["Aaron Marburg"]
  s.email       = ["aaron.marburg@pg.canterbury.ac.nz"]
  s.homepage    = "http://github.com/amarburg/opencv-ffi-ext"
  s.summary     = %q{Native compiled extensions to OpenCV-FFI.}
  s.description = %q{Native compiled extensions to OpenCV-FFI.}

  s.files         = `git ls-files`.split("\n")
#  s.extensions    << "ext/eigen/mkrf_conf.rb"
#  s.extensions    << "ext/opencv-ffi/mkrf_conf.rb"
#  s.extensions    << "ext/opensurf/mkrf_conf.rb"
  
  s.extensions << "ext/Rakefile"

  s.test_files    = `git ls-files -- {test,spec,features}/*`.split("\n")
  s.executables   = `git ls-files -- bin/*`.split("\n").map{ |f| File.basename(f) }
  s.require_paths = ["lib"]

  s.has_rdoc = true

  s.add_dependency "ffi"
  s.add_dependency "mkrf"
end
