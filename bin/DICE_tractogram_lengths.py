#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
import numpy as np
from tqdm import trange
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui
from dicelib.tractogram.processing import streamline_length

DESCRIPTION = """Return the length in mm of the streamlines in a tractogram"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_tractogram", help="Input tractogram")
    parser.add_argument("output_scalar_file", help="Output scalar file that will contain the streamline lengths")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwriting of the output")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser

def main():
    #----- check input -----
    parser = input_parser()
    options = parser.parse_args()
    if not os.path.isfile(options.input_tractogram):
        ui.ERROR( f'File "{options.input_tractogram}" not found' )
    if os.path.isfile(options.output_scalar_file) and not options.force:
        ui.ERROR( 'Output scalar file already exists, use -f to overwrite' )
    
    #----- iterate over input streamlines -----
    TCK_in = None
    try:
        # open the input file
        TCK_in = LazyTCK( options.input_tractogram, mode='r' )
        
        n_streamlines = int( TCK_in.header['count'] )
        if options.verbose:
            if n_streamlines>0:
                ui.INFO( f'{n_streamlines} streamlines in input tractogram' )
            else:
                ui.WARNING( 'The tractogram is empty' )

        lengths = np.empty( n_streamlines, dtype=np.double )
        if n_streamlines>0:
            for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading

                lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
        np.savetxt( options.output_scalar_file, lengths, fmt='%.4f' )

        if options.verbose and n_streamlines>0:
            ui.INFO( f'min={lengths.min():.3f}   max={lengths.max():.3f}   mean={lengths.mean():.3f}   std={lengths.std():.3f}' )
                

    except BaseException as e:
        if os.path.isfile( options.output_scalar_file ):
            os.remove( options.output_scalar_file )
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        if TCK_in is not None:
            TCK_in.close()


if __name__ == "__main__":
    main()