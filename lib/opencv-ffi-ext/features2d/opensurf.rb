
require 'nice-ffi'
require 'base64'

require 'opencv-ffi-wrappers/features2d/surf'

module CVFFI
  module OpenSURF
    extend NiceFFI::Library

    libs_dir = File.dirname(__FILE__) + "/../../../ext/opensurf/"
    pathset = NiceFFI::PathSet::DEFAULT.prepend( libs_dir )
    load_library("cvffi_opensurf", pathset)

    class OpenSURFParams < NiceFFI::Struct
      layout :upright, :char,
             :octaves, :int,
             :intervals, :int,
             :init_sample, :int,
             :thres, :float 
    end

    class OpenSURFPoint < NiceFFI::Struct
      @@descriptor_length = 64

      layout :pt, CvPoint,
        :scale, :float,
        :orientation, :float,
        :laplacian, :int,
        :descriptor, [ :float, @@descriptor_length ]

      def x;  pt.x; end
      def y;  pt.y; end

      def self.keys
        [ :scale, :orientation, :laplacian ]
      end
      def keys; self.class.keys; end

      def self.num_keys
        # Extra 3 for x,y,descriptor
        keys.length + 3
      end
      def num_keys; self.class.num_keys; end

      def to_a
        a = [ pt.x, pt.y ] + keys.map { |key| self[key] }
        a.push Base64.encode64( descriptor.to_a.pack( "g#{@@descriptor_length}" ) )
        a
      end

      def self.from_a(a)
        raise "Not enough elements in array to unserialize (#{a.length} < #{num_keys}" unless a.length == num_keys

        feature = OpenSURFPoint.new
        feature.pt.x = a.shift
        feature.pt.y = a.shift
        keys.each { |key|
          feature[key] = a.shift
        }
        desc = Base64.decode64(a.shift).unpack("g#{@@descriptor_length}")
        @@descriptor_length.times { |j| feature[:descriptor][j] = desc[j] }

        feature
      end

      def ==(b)
        result = keys.reduce( true ) { |m,s|
            puts "Key #{s} doesn't match" unless self[s] == b[s]
            m = m and (self[s] == b[s])
          } and ( pt.x == b.pt.x ) and ( pt.y == b.pt.y )

          result = ( result and (descriptor.to_a == b.descriptor.to_a) )
          result
        end
 
      def to_vector
        Vector.[]( x, y, 1 )
      end
      
      def to_Point
        pt.to_Point
      end

    end


    # CvSeq *opensurfDet( IplImage *img,
    #                   CvMemStorage *storage,
    #                   CvSURFParams params )
    attach_function :openSurfDetect, [ :pointer, :pointer, OpenSURFParams.by_value ], CvSeq.typed_pointer 
    attach_function :openSurfDescribe, [ :pointer, :pointer, OpenSURFParams.by_value ], CvSeq.typed_pointer 
    attach_function :createOpenSURFPointSequence, [:pointer ], CvSeq.typed_pointer

    class Results < SequenceArray
      sequence_class  OpenSURFPoint

      def mark_on_image( img, opts )
        each { |point|
          CVFFI::draw_circle( img, point.pt, opts )
        }
      end

    end

    class Params
      DEFAULTS = { upright: 0,
                   octaves: 5,
                   intervals: 4,
                   thres: 0.0004,
                   init_sample: 2 }

      def initialize( opts = {} )
        @params = {}
        DEFAULTS.keys.each { |k|
          @params[k] = opts[k] || DEFAULTS[k]
        }
      end

      def to_OpenSurfParams
        OpenSURFParams.new( @params )
      end

      def to_hash
        @params
      end
    end


    # Detection sets x,y,scale, laplacian
    def self.detect( img, params )
      params = params.to_OpenSurfParams unless params.is_a?( OpenSURFParams ) 
      raise ArgumentError unless params.is_a?( OpenSURFParams ) 

      mem_storage = CVFFI::cvCreateMemStorage( 0 )

      img = img.ensure_greyscale
      kp = CVFFI::CvSeq.new openSurfDetect( img, mem_storage, params )

      Results.new( kp, mem_storage )
    end

    # Descriptor takes x,y, scale.  Apparently not laplcian
    # Sets orientation, descriptor
    def self.describe( img, points, params )
      params = params.to_OpenSurfParams unless params.is_a?( OpenSURFParams ) 
      raise ArgumentError unless params.is_a?( OpenSURFParams ) 

      img = img.ensure_greyscale

      puts "Extracting #{points.length} features"

      kp = points.to_CvSeq

      openSurfDescribe( img, kp, params )

      points.reset(kp)
      points
    end

  end
end
