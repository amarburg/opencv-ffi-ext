
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
    @descriptors_one = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (10*i)+rand } }
    @descriptors_two = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (10*i)+rand } }.reverse

    @dmat_one = Mat.build( @num_descriptors, @dlength ) { |i,j| @descriptors_one[i][j] }
    @dmat_two = Mat.build( @num_descriptors, @dlength ) { |i,j| @descriptors_two[i][j] }
  end

  def test_brute_force_matcher
    [1,3,5].each { |k|
    Matcher::valid_norms.each { |norm|
      puts "Testing brute force matcher with norm #{norm}, k = #{k}"
      opts = { norm: norm }
      opts.merge!( knn: k ) if k > 1

      results = Matcher::brute_force_matcher( @dmat_one, @dmat_two, opts )

      assert_equal results.length, @num_descriptors*k

      case k
      when 1
        results.each { |result|
          assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
        }
      end
    }
    }
  end
end
