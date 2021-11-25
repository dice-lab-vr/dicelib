#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
import numpy as np
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui

DESCRIPTION = """Manipulate the streamlines in a tractogram"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("--weights_in", help="Text file with the input streamline weights")
    parser.add_argument("--minweight", type=float, default=0.0, help="Keep streamlines with weight >= this value")
    parser.add_argument("--maxweight", type=float, default=np.inf, help="Keep streamlines with weight <= this value")
    parser.add_argument("--weights_out", help="Text file for the output streamline weights")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser

def main():
    parser = input_parser()
    options = parser.parse_args()

    # check input
    if not os.path.isfile(options.input_tractogram):
        ui.ERROR( f'File "{options.input_tractogram}" not found' )
    if os.path.isfile(options.output_tractogram) and not options.force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )
    
    #----- actual code -----

    # read the streamline weights (if any)
    if options.weights_in is not None:
        weights = np.loadtxt( options.weights_in )
        if options.verbose:
            ui.LOG( 'Using streamline weights from text file' )
    else:
        weights = np.array( [] )
        if options.verbose:
            ui.LOG( 'Not using streamline weights' )

    # open the files
    TCK_in  = LazyTCK( options.input_tractogram, read_mode=True )
    TCK_out = LazyTCK( options.output_tractogram, read_mode=False, header=TCK_in.header )

    if 'count' in TCK_in.header.keys():
        n_streamlines = int( TCK_in.header['count'] )
        if options.verbose:
            ui.LOG( f'{n_streamlines} streamlines in input tractogram' )
    else:
        # TODO: allow the possibility to wotk also in this case
        ui.ERROR( '"count" field not found in header' )

    kept = np.ones( n_streamlines, dtype=bool )
    for i in range(n_streamlines):
        TCK_in.read_streamline()
        if TCK_in.n_pts==0:
            break # no more data, stop reading
        
        # filter out by weight
        if weights.size>0:
            if options.minweight and weights[i]<options.minweight:
                kept[i] = False
                continue
            if options.maxweight and weights[i]>options.maxweight:
                kept[i] = False
                continue
        TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )
    
    n_wrote = np.count_nonzero( kept )
    TCK_in.close()
    TCK_out.close( n_wrote )
    
    if weights.size>0 and options.weights_out is not None:
        print( weights[kept==True].size )
        np.savetxt( options.weights_out, weights[kept==True], fmt='%.5e' )

    if options.verbose:
        ui.LOG( f'{n_wrote} streamlines in output tractogram' )

if __name__ == "__main__":
    main()