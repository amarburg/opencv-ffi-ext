
require 'test/setup'
require 'opencv-ffi-wrappers'
require 'opencv-ffi-ext'

class TestMatcher < Test::Unit::TestCase
  include CVFFI

  def setup
    @dlength = 128
    @num_descriptors = 5

    # These descriptors should pretty unambiguously line up in reverse order 
    # (0 -> 4, 1-> 3, etc)
    @descriptors_one = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (10*i)+rand } }
    @descriptors_two = Array.new( @num_descriptors ) { |i| Array.new( @dlength ) { |j| (10*i)+rand } }.reverse

    @dmat_one = Mat.build( @num_descriptors, @dlength, {type: :CV_32F} ) { |i,j| @descriptors_one[i][j] }
    @dmat_two = Mat.build( @num_descriptors, @dlength, {type: :CV_32F} ) { |i,j| @descriptors_two[i][j] }
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

  # TODO:  BruteForceRadius doesn't appear to be working...
  def test_brute_force_ratio_test
    [2.0].each { |ratio|
      Matcher::valid_norms.each { |norm|
        puts "Testing brute force matcher with norm #{norm}, ratio = #{ratio}"
        opts = { norm: norm, ratio: ratio }

        results = Matcher::brute_force_matcher( @dmat_one, @dmat_two, opts )

        puts "   ... returned #{results.length} matches"
        #assert_equal @num_descriptors, results.length

        #results.each { |result|
        #  assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
        #}
      }
    }
  end

  # TODO:  BruteForceRadius doesn't appear to be working...
  def test_brute_force_radius
    [5.0].each { |radius|
      Matcher::valid_norms.each { |norm|
        puts "Testing brute force matcher with norm #{norm}, radius = #{radius}"
        opts = { norm: norm, radius: radius }

        results = Matcher::brute_force_matcher( @dmat_one, @dmat_two, opts )

        #assert_equal @num_descriptors, results.length

        #results.each { |result|
        #  assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
        #}
      }
    }
  end

  # TODO:  Currently, the matching API is focused on matching image pairs, not on training...
  def test_flann_based_matcher_knn
    [1,3,5].each { |k|
      puts "Testing flann-based matcher with k = #{k}"
      opts = (k>1) ?  { knn: k } : {}

      results = Matcher::flann_based_matcher( @dmat_one, @dmat_two, opts )

      assert_equal results.length, @num_descriptors*k

      case k
      when 1
        results.each { |result|
          assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
        }
      end
    }
  end

  def test_flann_based_matcher_radius
    [100.0].each { |radius|
      puts "Testing flann-based matcher with radius = #{radius}"
      opts = {radius: radius }

      results = Matcher::flann_based_matcher( @dmat_one, @dmat_two, opts )

      assert_equal results.length, @num_descriptors

      results.each { |result|
        assert_equal (result.queryIdx + result.trainIdx), (@num_descriptors-1)
      }
    }
  end
end
