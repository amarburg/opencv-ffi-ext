source "http://rubygems.org"
gem "rake"

# Specify your gem's dependencies in opencv-ffi.gemspec
gemspec

# Uses my version of nice-ffi which allows for null constructors 
# to structs

def my_github( x ); "http://github.com/amarburg/#{x}.git"; end

gem "nice-ffi", :git=>my_github("nice-ffi")

#gem 'opencv-ffi', :git=>my_github("opencv-ffi")
gem 'opencv-ffi', :path=>"../opencv-ffi"

gem "mkrf"

group :development do
  gem "redcarpet"
  gem "simplecov", :require => false
  gem "yard"
end

