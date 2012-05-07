source "http://rubygems.org"
gem "rake"

# Specify your gem's dependencies in opencv-ffi.gemspec
gemspec

# Uses my version of nice-ffi which allows for null constructors 
# to structs
gem "nice-ffi", :git=>"git@github.com:amarburg/nice-ffi.git"

group :development do
  gem "redcarpet"
  gem "simplecov", :require => false
  gem "yard"
end

