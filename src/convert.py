from Environment import *
import pickle
import collections

def convert_option( o ):
    # Detect need for conversion
    o.pi = dict( [ (s,((a,pr),)) for (s,(a,pr)) in o.pi.items() ] )
    return o

def main(in_fname, out_fname):
    O = pickle.load( open( in_fname ) )
    O = map( convert_option, O )
    pickle.dump(O, open( out_fname, "w" ))

if __name__ == "__main__":
    import sys
    if len( sys.argv ) <> 3:
        print "Usage: %s <in> <out>"%( sys.argv[0] )
        sys.exit( 1 )

    in_fname = sys.argv[1]
    out_fname = sys.argv[2]

    main( in_fname, out_fname )

