
require 'opencv-ffi'
require 'opencv-ffi-wrappers/core/sequence'

module CVFFI

  module Matcher
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../ext/opencv-ffi/"
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

    attach_function :bruteForceMatcher, [:pointer, :pointer, :pointer, :int, :bool], CvSeq.typed_pointer
    attach_function :bruteForceMatcherKnn, [:pointer, :pointer, :pointer, :int, :int, :bool], CvSeq.typed_pointer
    attach_function :bruteForceMatcherRadius, [:pointer, :pointer, :pointer, :int, :float, :bool], CvSeq.typed_pointer

    def self.valid_norms
      [ :NORM_L1, :NORM_L2, :NORM_L2SQR ]   
    end

    def self.brute_force_matcher( query, train, opts = {} )
      normType = opts[:norm_type] || opts[:norm] || :NORM_L2
      knn = opts[:knn] || 1
      radius = opts[:radius] || nil
      crossCheck = false 

      pool = CVFFI::cvCreateMemStorage(0);
      seq = if radius.nil?
              bruteForceMatcherKnn( query.to_CvMat, train.to_CvMat, pool, normType, knn, crossCheck )
            else
              bruteForceMatcherRadius( query.to_CvMat, train.to_CvMat, pool, normType, radius, crossCheck )
            end

      MatchResults.new( seq, pool );
    end

    class DMatch < NiceFFI::Struct
      layout  :queryIdx, :int,
              :trainIdx, :int,
              :imgIdx, :int,
              :distance, :float
    end

    class MatchResults < SequenceArray
      sequence_class DMatch
    end
  end


end
