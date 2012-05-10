source "http://rubygems.org"
gem "rake"

# Specify your gem's dependencies in opencv-ffi.gemspec
gemspec

# Uses my version of nice-ffi which allows for null constructors 
# to structs

def my_github( x ); "http://github.com/amarburg/#{x}.git"; end

gem "nice-ffi", :git=>my_github("nice-ffi")

if ENV["DEV"].to_i > 0
  gem 'opencv-ffi', :path=>"../opencv-ffi"
else
  gem 'opencv-ffi', :git=>my_github("opencv-ffi"), :branch=>"no-ext"
end

group :development do
  gem "redcarpet"
  gem "simplecov", :require => false
  gem "yard"
end

