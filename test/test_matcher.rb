
require 'test/setup'
require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext/matcher'

class TestMatcher < Test::Unit::TestCase
  include CVFFI

  def setup
    @dlength = 128
    @num_descriptors = 5

    # These descriptors should pretty unambiguously line up in reverse order 
    # (0 -> 4, 1-> 3, etc)
    @descriptors_one = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (2*i)+rand } }
    @descriptors_two = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (2*i)+rand } }.reverse

    @dmat_one = Mat.build( @num_descriptors, @dlength ) { |i,j| @descriptors_one[i][j] }
    @dmat_two = Mat.build( @num_descriptors, @dlength ) { |i,j| @descriptors_two[i][j] }
  end

  def test_brute_force_matcher

    results = Matcher::brute_force_matcher( @dmat_one, @dmat_two )

    results.each { |result|
      p result
      assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
    }
  end
end
