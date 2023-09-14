#!python
# cython: boundscheck=False, wraparound=False, profile=False, language_level=3

import os
import numpy as np
cimport numpy as np
import nibabel as nib
import os, random as rnd
from dicelib.lazytractogram cimport LazyTractogram
from dicelib.ui import ProgressBar
from dicelib import ui

cpdef split_clusters(tractogram, clust_idx, output_folder, verbose=3):
    TCK_in          = None
    TCK_out         = None
    TCK_outs        = {}
    TCK_outs_size   = {}
    max_open = int( os.sysconf('SC_OPEN_MAX')*0.9 )
    n_written         = 0
    try:
        # open the tractogram
        TCK_in = LazyTractogram( tractogram, mode='r' )
        n_streamlines = int( TCK_in.header['count'] )
        ui.INFO( f'{n_streamlines} streamlines in input tractogram' )

        if n_streamlines!=clust_idx.shape[0]:
            ui.ERROR( f'# of indexes ({clust_idx.shape[0]}) is different from # of streamlines ({n_streamlines}) ' )
        # check if #(weights)==n_streamlines

        # create empty tractograms for unique assignments
        unique_assignments = np.unique(clust_idx, axis=0)
        ui.INFO(f"number of clusters: {unique_assignments.size}")

        for i in range( unique_assignments.shape[0] ):
            key = f'{unique_assignments[i]}'

            TCK_outs[key] = None
            TCK_outs_size[key] = 0
            tmp = LazyTractogram( os.path.join(output_folder,f'{key}.tck'), mode='w', header=TCK_in.header )
            tmp.close( write_eof=False, count=0 )

        ui.INFO( f'Created {len(TCK_outs)} empty files for output tractograms' )

        #----  iterate over input streamlines  -----
        n_file_open = 0
        with ProgressBar( total=n_streamlines, disable=(verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
            for i in range(n_streamlines):
                TCK_in.read_streamline()
                if TCK_in.n_pts==0:
                    break # no more data, stop reading
                # get the key of the dictionary
                key = f'{clust_idx[i]}'

                # check if need to open file
                if TCK_outs[key] is None:
                    fname = os.path.join(output_folder,f'{key}.tck')
                    if n_file_open==max_open:
                        key_to_close = rnd.choice( [k for k,v in TCK_outs.items() if v!=None] )
                        TCK_outs[key_to_close].close( write_eof=False )
                        TCK_outs[key_to_close] = None
                    else:
                        n_file_open += 1

                    TCK_outs[key] = LazyTractogram( fname, mode='a' )

                # write input streamline to correct output file
                TCK_outs[key].write_streamline( TCK_in.streamline, TCK_in.n_pts )
                TCK_outs_size[key] += 1
                n_written += 1
                pbar.update()
    except Exception as e:
        ui.ERROR(e)

    finally:
        ui.INFO( 'Closing files' )
        if TCK_in is not None:
            TCK_in.close()
        for key in TCK_outs.keys():
            f = os.path.join(output_folder,f'{key}.tck')
            if not os.path.isfile(f):
                continue
            if TCK_outs[key] is not None:
                TCK_outs[key].close( write_eof=False )
            # Update 'count' and write EOF marker
            tmp = LazyTractogram( f, mode='a' )
            tmp.close( write_eof=True, count=TCK_outs_size[key] )