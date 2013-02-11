
require 'opencv-ffi'
require 'opencv-ffi-wrappers/core/sequence'

module CVFFI

  module Matcher
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    NormTypes = enum :norm_types, [ :NORM_INF, 1,
                                    :NORM_L1, 2,
                                    :NORM_L2, 4,
                                    :NORM_L2SQR, 5,
                                    :NORM_HAMMING, 6,
                                    :NORM_HAMMING2, 7,
                                    :NORM_TYPE_MASK, 7,
                                    :NORM_RELATIVE, 8,
                                    :NORM_MINMAX, 32 ]

    def self.valid_norms
      [ :NORM_L1, :NORM_L2, :NORM_L2SQR ]   
    end


    class CvMatcherParams  < NiceFFI::Struct
      layout :norm_type, :int,
             :knn, :int,
             :calculateRatios, :bool,
             :ratio, :float,
             :radius, :float,
             :crossCheck, :bool

      def to_CvMatcherParams; self; end
    end

    class MatcherParams < CVFFI::Params 
      param :norm_type, NormTypes[:NORM_L2]
      param :knn, 1
      param :calculateRatios, false
      param :ratio, 0.0
      param :radius, 0.0
      param :crossCheck, false

      def to_CvMatcherParams
        CvMatcherParams.new( @params )
      end
    end


    # Brute force matcher
    #
    attach_function :bruteForceMatcherParams, [:pointer, :pointer, :pointer, :pointer], CvSeq.typed_pointer
    attach_function :bruteForceMatcher, [:pointer, :pointer, :pointer, :int, :bool], CvSeq.typed_pointer
    attach_function :bruteForceMatcherKnn, [:pointer, :pointer, :pointer, :int, :int, :bool], CvSeq.typed_pointer
    attach_function :bruteForceMatcherRadius, [:pointer, :pointer, :pointer, :int, :float, :bool], CvSeq.typed_pointer
    attach_function :bruteForceMatcherRatioTest, [:pointer, :pointer, :pointer, :int, :float, :bool], CvSeq.typed_pointer

    def self.brute_force_matcher( query, train, opts = {} )
      # Translate some of the params
      params = MatcherParams.new( opts )

      pool = CVFFI::cvCreateMemStorage(0);
      seq =  bruteForceMatcherParams( query.to_CvMat, train.to_CvMat, pool, params.to_CvMatcherParams )

      MatchResults.new( seq, pool );
    end

    # Flann-based matcher
    #
    attach_function :flannBasedMatcher, [:pointer, :pointer, :pointer], CvSeq.typed_pointer
    attach_function :flannBasedMatcherKnn, [:pointer, :pointer, :pointer, :int ], CvSeq.typed_pointer
    attach_function :flannBasedMatcherRadius, [:pointer, :pointer, :pointer, :float ], CvSeq.typed_pointer
    attach_function :flannBasedMatcherRatioTest, [:pointer, :pointer, :pointer, :float ], CvSeq.typed_pointer

    def self.flann_based_matcher( query, train, opts = {} )
      knn = opts[:knn] || 1
      radius = opts[:radius] || nil

      pool = CVFFI::cvCreateMemStorage(0);
      seq = if radius.nil?
              flannBasedMatcherKnn( query.to_CvMat, train.to_CvMat, pool, knn )
            else
              flannBasedMatcherRadius( query.to_CvMat, train.to_CvMat, pool, radius )
            end

      MatchResults.new( seq, pool );
    end

    # Match results
    #
    # A DMatch is strictly index based (doesn't store the actual X,Y 
    # coords of the points).
    # It's typically stored in a MatchResults (a wrapper around a CvSeq)
    #
    # For that see the (Ruby, not OpenCV) Matches class below
    #
    class DMatch < NiceFFI::Struct
      layout  :queryIdx, :int,
              :trainIdx, :int,
              :imgIdx, :int,
              :distance, :float,
              :ratio, :float

      def self.keys
        [ :queryIdx, :trainIdx, :imgIdx, :distance, :ratio ]
      end

      def keys; self.class.keys; end

      def to_a
        keys.map { |key| send key }
      end

      def self.from_a(arr)
        raise "Incorrect number of elements in array -- it's #{arr.length}, expecting #{keys.length}" unless arr.length == keys.length
        h = {}
        keys.each { |key| h[key] = arr.shift }
        DMatch.new( h )
      end
    end

    class MatchResults < SequenceArray
      sequence_class DMatch
    end

   #
    # Small abstraction breakage.  The data in Mogile is a serialization of
    # a CVFFI struct.   Why am I recreating the data in a different class
    # instead of just recreating the OpenCV class?
    class Matches
      include Enumerable

      def initialize( matches )
        @matches = matches
      end

      def length; @matches.length; end

      def at(i)
        @matches[i]
      end
      alias :[] :at

      def each
        length.times { |i| yield at(i) }
      end

      def first( n = 1 )
        Matches.new( @matches.first(n) )
      end
        

      def to_CvMat( which )
        CVFFI::Mat.build( length, 2, :CV_32F ) { |i,j|
          pt = @matches[i].which(which)
          (j == 0) ?  pt.x : pt.y
        }
      end

      def to_Points( which )
        @matches.map { |m| Point.new(  m.which( which ) ) }
      end

      def to_a
        map { |match| match.to_a } 
      end

      def self.from_DMatch( matches, query, train )
        Matches.new( Array.new( matches.length ) { |i|
          m = matches[i]
          q = query[m.queryIdx]
          t = train[m.trainIdx]
          Match.from_points( CVFFI::Point.new( q ), CVFFI::Point.new( t ), m.distance )
        } )
      end

      def self.unserialize( a )
        Matches.new( a.map! { |d|
          raise "In Matches.unserialize, line in array isn't the correct length, have #{d.length}, expected 5" unless d.length == 5
          Match.from_points( CVFFI::Point.new( d.shift(2) ), CVFFI::Point.new( d.shift(2) ), d.shift )
        } )
      end

    end



  end


end
