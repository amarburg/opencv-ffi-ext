
require 'opencv-ffi'
require 'opencv-ffi-wrappers/core/sequence'

module CVFFI

  module Matcher
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    attach_function :bruteForceMatcher, [:pointer, :pointer, :pointer, :int, :bool], CvSeq.typed_pointer

    def self.brute_force_matcher( query, train, normType = 0, crossCheck = false )
      
      pool = CVFFI::cvCreateMemStorage(0);

      seq = bruteForceMatcher( query.to_CvMat, train.to_CvMat, pool, normType, crossCheck )
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
