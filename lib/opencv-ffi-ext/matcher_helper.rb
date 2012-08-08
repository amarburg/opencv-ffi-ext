
require 'nice-ffi'
require 'opencv-ffi-wrappers/matcher'

module CVFFI
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../ext/opencv-ffi/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi", pathset)

    attach_function :computeReprojError, [:pointer, :pointer], :double
    attach_function :computeSetReprojError, [:pointer, :pointer], :void
    attach_function :computeSeqReprojError, [:pointer, :pointer, :pointer], :void

    def self.compute_reproj_error( match, model )
      computeReprojError( match, model.to_CvMat( :type => :CV_64F ) )
    end

    def self.compute_set_reproj_error( matchSet, model )
      computeSetReprojError( matchSet, model.to_CvMat( :type => :CV_64F ) ) 
    end

    def self.compute_seq_reproj_error( matchSeq, model )
      errors = FFI::MemoryPointer.new( :double, matchSeq.length )
      computeSeqReprojError( matchSeq, model.to_CvMat( :type => :CV_64F ), errors ) 
      errors.read_array_of_double
    end
end


