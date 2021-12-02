#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys
import numpy as np
from tqdm import trange
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui
from dicelib.tractogram.processing import streamline_length

DESCRIPTION = """Print some information about a tractogram"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("tractogram", help="Input tractogram")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser

def main():
    #----- check input -----
    parser = input_parser()
    options = parser.parse_args()
    if not os.path.isfile(options.tractogram):
        ui.ERROR( f'File "{options.tractogram}" not found' )
    
    #----- iterate over input streamlines -----
    TCK_in  = None
    try:
        # open the input file
        TCK_in = LazyTCK( options.tractogram, mode='r' )

        # print the header
        ui.INFO( 'HEADER content')
        max_len = max([len(k) for k in TCK_in.header.keys()])
        for k, v in TCK_in.header.items():
            print( ui.hWhite+ '%0*s'%(max_len,k) +ui.Reset+ui.fWhite+ ':  ' + v +ui.Reset )
        print( '' )

        # print stats on lengths
        ui.INFO( 'Streamline lenghts')
        n_streamlines = int( TCK_in.header['count'] )
        lengths = np.empty( n_streamlines, dtype=np.double )
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            lengths[i] = streamline_length( TCK_in.streamline, TCK_in.n_pts )
        print( f'   {ui.hWhite}min{ui.Reset}{ui.fWhite}={lengths.min():.3f}   {ui.hWhite}max{ui.Reset}{ui.fWhite}={lengths.max():.3f}   {ui.hWhite}mean{ui.Reset}{ui.fWhite}={lengths.mean():.3f}   {ui.hWhite}std{ui.Reset}{ui.fWhite}={lengths.std():.3f}{ui.Reset}' )

    except Exception as e:
        ui.ERROR( e.__str__() )

    finally:
        if TCK_in is not None:
            TCK_in.close()


if __name__ == "__main__":
    main()