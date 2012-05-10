
require 'opencv-ffi/cvffi'
require 'opencv-ffi/core'
require 'opencv-ffi/features2d/library'
require 'opencv-ffi-ext/misc.rb'

module CVFFI

  module Features2D

    class CvKeyPoint < NiceFFI::Struct
      layout :x, :float,
        :y, :float,
        :kp_size, :float,
        :angle, :float,
        :response, :float,
        :octave, :int

      def ==(b)
        members.reduce(true) { |memo, member| memo and self[member] == b[member] }
      end
      def ===(b)
        members.reduce(true) { |memo, member| memo and self[member] === b[member] }
      end


      def to_a
        values
      end

      def self.from_a( a )
        raise "Wrong number of elements" unless a.length == members.length
        CvKeyPoint.new( a )
      end
    end

    class CvEllipticKeyPoint < NiceFFI::Struct
      layout :centre, CvPoint,
        :axes, CvSize2D32f,
        :phi, :double,
        :kp_size, :float,
        :si, :float,
        :transf, CvMat

      def x; centre.x; end
      def y; centre.y; end

      def ==(b)
        members.reduce(true) { |memo, member| 
          #puts "Member #{member} doesn't match #{self[member].to_s}, #{self[member].to_s}" unless self[member] == b[member]
          memo and (member == :transf ? true : self[member] == b[member] )
        }
      end

      def self.keys
        [ :phi, :kp_size, :si ]
      end
      def keys; CvEllipticKeyPoint.keys; end

      def to_a
        [ centre.x, centre.y, axes.x, axes.y ] + keys.map { |k| self[k] }
      end

      def self.from_a( a )
        raise "Wrong number of elements" unless a.length == (4 + keys.length)
        feature = CvEllipticKeyPoint.new( nil )
        feature.centre.x = a.shift
        feature.centre.y = a.shift
        feature.axes.x = a.shift
        feature.axes.y = a.shift
        keys.each { |k|
          feature[k] = a.shift
        }
        feature
      end
    end

    class Keypoints < SequenceArray
      sequence_class CvKeyPoint 
    end

    class EllipticKeypoints < SequenceArray
      sequence_class CvEllipticKeyPoint 
    end

  end
end
