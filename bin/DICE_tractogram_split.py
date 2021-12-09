#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, sys, glob, random
import numpy as np
from tqdm import trange
from dicelib.tractogram.lazytck import LazyTCK
import dicelib.ui as ui

DESCRIPTION = """Split the streamlines in a tractogram according to a (precomputed) assignment file"""

def input_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("tractogram", help="Input tractogram")
    parser.add_argument("assignments", help="Text file with the streamline assignments")
    parser.add_argument("output_folder", help="Output folder for the splitted tractograms")
    parser.add_argument("--max_open", "-m", type=int, default=128, help="Maximum number of files open at the same time")
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

    ui.set_verbose( 2 if options.verbose else 1 )

    # check input
    if not os.path.isfile(options.tractogram):
        ui.ERROR( f'File "{options.tractogram}" not found' )
    if not os.path.isfile(options.assignments):
        ui.ERROR( f'File "{options.assignments}" not found' )
    if not os.path.isdir(options.output_folder):
        os.mkdir( options.output_folder )
    else:
        if options.force:
            for f in glob.glob( os.path.join(options.output_folder,'*.tck') ):
                os.remove(f)
        else:
            ui.ERROR( 'Output folder already exists, use -f to overwrite' )

    #----- iterate over input streamlines -----
    TCK_outs      = {}
    TCK_outs_size = {}
    n_wrote       = 0
    try:
        # open the tractogram
        TCK_in = LazyTCK( options.tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        # open the assignments
        assignments = np.loadtxt( options.assignments, dtype=int )
        if assignments.ndim!=2 or assignments.shape[1]!=2:
            ui.ERROR( 'Unable to open assignments file' )
        ui.INFO( f'{assignments.shape[0]} assignments in input text file' )

        # check if #(assignments)==n_streamlines
        if n_streamlines!=assignments.shape[0]:
            ui.ERROR( f'# of assignments ({assignments.shape[0]}) is different from # of streamlines ({n_streamlines}) ' )

        # create empty tractograms for unique assignments
        unique_assignments = np.unique(assignments, axis=0)
        for i in range( unique_assignments.shape[0] ):
            if unique_assignments[i,0]==0 or unique_assignments[i,1]==0:
                continue
            if unique_assignments[i,0] <= unique_assignments[i,1]:
                key = f'{unique_assignments[i,0]}-{unique_assignments[i,1]}'
            else:
                key = f'{unique_assignments[i,1]}-{unique_assignments[i,0]}'
            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTCK( os.path.join(options.output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )
        # TCK_outs['nc'] = None # add key for non-connecting streamlines
        ui.INFO( f'Created {len(TCK_outs)} empty files for output tractograms' )

        #----  iterate over input streamlines  -----
        n_file_open = 0
        for i in trange( n_streamlines, bar_format='{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=False ):
            TCK_in.read_streamline()
            if TCK_in.n_pts==0:
                break # no more data, stop reading
            # get the key of the dictionary
            if assignments[i,0]==0 or assignments[i,1]==0:
                key = 'unassigned'
            elif assignments[i,0] <= assignments[i,1]:
                key = f'{assignments[i,0]}-{assignments[i,1]}'
            else:
                key = f'{assignments[i,1]}-{assignments[i,0]}'

            # check if need to open file
            if TCK_outs[key] is None:
                fname = os.path.join(options.output_folder,f'{key}.tck')
                if n_file_open==options.max_open:
                    key_to_close = random.choice( [k for k,v in TCK_outs.items() if v!=None] )
                    TCK_outs[key_to_close].close( write_eof=False )
                    TCK_outs[key_to_close] = None
                else:
                    n_file_open += 1

                TCK_outs[key] = LazyTCK( fname, mode='a' )

            # write input streamline to correct output file
            TCK_outs[key].write_streamline( TCK_in.streamline, TCK_in.n_pts )
            TCK_outs_size[key] += 1
            n_wrote += 1

        ui.INFO( f'{n_wrote} streamlines written in total' )

    except BaseException as e:
        ui.ERROR( e.__str__() if e.__str__() else 'A generic error has occurred' )

    finally:
        ui.INFO( 'Closing files and update totals' )
        if TCK_in is not None:
            TCK_in.close()
        for key in TCK_outs.keys():
            if TCK_outs[key] is not None:
                TCK_outs[key].close( write_eof=False )
            # Update 'count' and write EOF marker
            tmp = LazyTCK( os.path.join(options.output_folder,f'{key}.tck'), mode='a' )
            tmp.close( write_eof=True, count=TCK_outs_size[key] )


if __name__ == "__main__":
    main()