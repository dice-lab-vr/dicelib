#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
import numpy as np
from tqdm import trange
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui
from dicelib.tractogram.processing import streamline_length

DESCRIPTION = """Manipulate the streamlines in a tractogram"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_tractogram", help="Output tractogram")
    parser.add_argument("--minlength", type=float, help="Keep streamlines with length [in mm] >= this value")
    parser.add_argument("--maxlength", type=float, help="Keep streamlines with length [in mm] <= this value")
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
    
    n_wrote = 0
    TCK_in  = None
    TCK_out = None

    # check input
    if not os.path.isfile(options.input_tractogram):
        ui.ERROR( f'File "{options.input_tractogram}" not found' )
    if os.path.isfile(options.output_tractogram) and not options.force:
        ui.ERROR( 'Output tractogram already exists, use -f to overwrite' )
    
    if options.minlength is not None:
        if options.minlength<0:
            ui.ERROR( '"minlength" must be >= 0' )
        if options.maxlength is not None and options.minlength>options.maxlength:
            ui.ERROR( '"minlength" must be <= "maxlength"' )
        if options.verbose:
            ui.LOG( f'Discard streamlines with length < {options.minlength}' )
    if options.maxlength is not None:
        if options.maxlength<0:
            ui.ERROR( '"maxlength" must be >= 0' )
        if options.verbose:
            ui.LOG( f'Discard streamlines with length > {options.maxlength}' )

    # read the streamline weights (if any)
    if options.weights_in is not None:
        if not os.path.isfile( options.weights_in ):
            ui.ERROR( f'File "{options.weights_in}" not found' )
        weights = np.loadtxt( options.weights_in )
        if options.verbose:
            ui.LOG( 'Using streamline weights from text file' )
    else:
        weights = np.array( [] )

    #----- iterate over input streamlines -----
    try:
        # open the files
        TCK_in = LazyTCK( options.input_tractogram, read_mode=True )
        if 'count' in TCK_in.header.keys():
            n_streamlines = int( TCK_in.header['count'] )
            if options.verbose:
                ui.LOG( f'{n_streamlines} streamlines in input tractogram' )
        else:
            # TODO: allow the possibility to wotk also in this case
            ui.ERROR( '"count" field not found in header' )
        if options.weights_in and n_streamlines!=weights.size:
            ui.ERROR( f'# of weights {weights.size} is different from # of streamlines ({n_streamlines}) ' )

        TCK_out = LazyTCK( options.output_tractogram, read_mode=False, header=TCK_in.header )

        kept = np.ones( n_streamlines, dtype=bool )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading

            # filter by length
            if options.minlength is not None or options.maxlength is not None:
                length = streamline_length(TCK_in.streamline, TCK_in.n_pts)
                if options.minlength is not None and length<options.minlength :
                    kept[i] = False
                    continue
                if options.maxlength is not None and length>options.maxlength :
                    kept[i] = False
                    continue
            
            # # filter out by weight
            if options.weights_in is not None and (
                (options.minweight and weights[i]<options.minweight) or
                (options.maxweight and weights[i]>options.maxweight)
            ):
                    kept[i] = False
                    continue
            TCK_out.write_streamline( TCK_in.streamline, TCK_in.n_pts )

        if weights.size>0 and options.weights_out is not None:
            print( weights[kept==True].size )
            np.savetxt( options.weights_out, weights[kept==True], fmt='%.5e' )

        n_wrote = np.count_nonzero( kept )
        if options.verbose:
            ui.LOG( f'{n_wrote} streamlines in output tractogram' )

    except:
        if TCK_out is not None:
            TCK_out.close()
        if os.path.isfile( options.output_tractogram ):
            os.remove( options.output_tractogram )
        if options.weights_out and os.path.isfile( options.weights_out ):
            os.remove( options.weights_out )
        ui.ERROR( 'Unable to process streamlines in the tractogram' )

    finally:
        if TCK_in is not None:
            TCK_in.close()
        if TCK_out is not None:
            TCK_out.close( n_wrote )

if __name__ == "__main__":
    main()