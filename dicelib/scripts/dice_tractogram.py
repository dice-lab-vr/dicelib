from dicelib.clustering import run_clustering
from dicelib.connectivity import assign
from dicelib.tractogram import compute_lengths, filter as tract_filter, info, join as tract_join, recompute_indices, get_indices_of_streamlines, resample, sample, tsf_create, sanitize, spline_smoothing_v2, split, sort as tract_sort, tsf_join
from dicelib.ui import setup_logger, setup_parser

import os
from time import time

import numpy as np

logger = setup_logger('dice_tractogram')


def tractogram_assign():
    '''
    Entry point for the tractogram assignment function.
    '''
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['atlas'], {'type': str, 'help': 'Path to the atlas file used to compute streamlines assignments'}],
        [['assignments_out'], {'type': str, 'help': 'Output assignments file (.txt or .npy)'}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': '''\
                                  Distance used to perform a radial search from each streamline endpoint to locate the nearest node and assign the streamline to the corresponding bundle.
                                  Argument is the maximum radius in mm'''}],
        [['--n_threads', '-n'], {'type': int, 'default': 3, 'metavar': 'N_THREADS', 'help': '''\
                                 Number of threads to use to perform the assignment.
                                 If None, all the available threads will be used'''}]
    ]
    options = setup_parser(assign.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    assign(options.tractogram_in,
           options.atlas,
           options.assignments_out,
           options.atlas_dist,
           options.n_threads,
           options.force,
           options.verbose)


def tractogram_cluster():
    '''
    Entry point for the tractogram clustering function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['thr'], {'type': float, 'help': 'Distance threshold [in mm] used to cluster the streamlines'}],
        [['tractogram_out'], {'type': str, 'default': None, 'help': 'Output clustered tractogram'}],
        [['--metric', '-m'], {'type': str, 'default': 'mean', 'metavar': 'METRIC', 'help':'''\
                                            Metric used to cluster the streamlines. Options: \'mean\', \'max\'.
                                            If \'max\', streamlines with ALL the points closer than \'thr\' will be clustered together.
                                            If \'mean\', streamlines with AVERAGE distance closer than \'thr\' will be clustered together'''}],
        [['--n_pts', '-n'], {'type': int, 'default': 12, 'metavar': 'N_PTS', 'help': 'Resample all streamlines to N_PTS points. Clustering requires streamlines to have the same number of points'}],
        [['--atlas', '-a'], {'type': str, 'metavar': 'ATLAS_FILE', 'help': '''\
                                            Path to the atlas file used to split the streamlines into bundles and clustering each of them in parallel;
                                            if not provided, the clustering will be performed sequentially'''}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': '''\
                                            Distance used to perform a radial search from each streamline endpoint to locate the nearest node and assign the streamline to the corresponding bundle.
                                            Argument is the maximum radius in mm; if no node is found within this radius, the streamline is not taken into account for clustering'''}],
        [['--weights_in', '-w_in'], {'type': str, 'default': None, 'help': 'Text file containing a scalar value for each streamline used to assign a weight to the final centroid of each cluster'}],
        [['--weights_out', '-w_out'], {'type': str, 'default': None, 'help': 'Text file for the output streamline weights'}],
        [['--weights_metric', '-w_m'], {'type': str, 'default': 'sum', 'metavar': 'WEIGHTS_METRIC', 'help': '''\
                                            Metric used to compute the final weight of each cluster centroid. Options: \'sum\', \'mean\', \'max\', \'median\', \'min\'.
                                            If \'sum\', the final weight is the sum of all the weights of the streamlines in the cluster.
                                            If \'mean\', the final weight is the mean of all the weights of the streamlines in the cluster.
                                            If \'max\', the final weight is the maximum of all the weights of the streamlines in the cluster.
                                            If \'median\', the final weight is the median of all the weights of the streamlines in the cluster.
                                            If \'min\', the final weight is the minimum of all the weights of the streamlines in the cluster'''}],
        [['--tmp_folder', '-tmp'], {'type': str, 'default': 'tmp', 'metavar': 'TMP_FOLDER', 'help': 'Path to the temporary folder used to store the intermediate files for parallel clustering'}],
        [['--max_open_files'], {'type': int, 'default': None, 'metavar': 'MAX_OPEN_FILES', 'help': 'Maximum number of files opened at the same time used to split the streamlines into bundles for parallel clustering'}],
        [['--n_threads'], {'type': int, 'metavar': 'N_THREADS', 'help': 'Number of threads to use to perform parallel clustering. If None, all the available threads will be used'}],
        [['--keep_temp', '-k'], {'action': 'store_true', 'help': 'Keep temporary files'}]
    ]
    options = setup_parser(run_clustering.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    run_clustering(
        tractogram_in=options.tractogram_in,
        temp_folder=options.tmp_folder,
        tractogram_out=options.tractogram_out,
        atlas=options.atlas,
        conn_thr=options.atlas_dist,
        clust_thr=options.thr,
        metric=options.metric,
        n_pts=options.n_pts,
        weights_in=options.weights_in,
        weights_metric=options.weights_metric,
        weights_out=options.weights_out,        
        n_threads=options.n_threads,
        force=options.force,
        verbose=options.verbose,
        keep_temp_files=options.keep_temp,
        max_open=options.max_open_files
    )


# def tractogram_compress():
#     # parse the input parameters
#     args = [
#         [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
#         [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
#         [['--minlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] >= this value'}],
#         [['--maxlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] <= this value'}],
#         [['--minweight'], {'type': float, 'help': 'Keep streamlines with weight >= this value'}],
#         [['--maxweight'], {'type': float, 'help': 'Keep streamlines with weight <= this value'}],
#         [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights'}],
#         [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights'}]
#     ]
#     options = setup_parser('Not implemented', args, add_force=True, add_verbose=True)

#     logger.error('This function is not implemented yet')


# def tractogram_convert():
#     set_sft_logger_level("CRITICAL")
#     args = [
#         [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
#         [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
#         [['--reference', '-r'], {'type': str, 'help': 'Space attributes used as reference for the input tractogram'}],
#         [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}]
#     ]
#     options = setup_parser("Tractogram conversion from and to '.tck', '.trk', '.fib', '.vtk' and 'dpy'. All the extensions except '.trk, need a NIFTI file as reference", args)

#     if not os.path.isfile(options.tractogram_in):
#         ERROR("No such file {}".format(options.tractogram_in))
#     if os.path.isfile(options.tractogram_out) and not options.force:
#         ERROR("Output tractogram already exists, use -f to overwrite")
#     if options.reference is not None:
#         if not os.path.isfile(options.reference):
#             ERROR("No such file {}".format(options.reference))

#     if not options.tractogram_in.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
#         ERROR("Invalid input tractogram format")
#     elif not options.tractogram_out.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
#         ERROR("Invalid input tractogram format")
#     elif options.reference is not None and not options.reference.endswith(('.nii', 'nii.gz')):
#         ERROR("Invalid reference format")

#     if options.tractogram_in.endswith('.tck') and options.reference is None:
#         ERROR("Reference is required if the input format is '.tck'")

#     try:
#         sft_in = load_tractogram(
#             options.tractogram_in,
#             reference=options.reference if options.reference else "same"
#         )
#     except Exception:
#         raise ValueError("Error loading input tractogram")
    
#     try:
#         save_tractogram(sft_in, options.tractogram_out)
#     except (OSError, TypeError) as e:
#         ERROR(f"Output not valid: {e}")


def tractogram_create_tsf():
    '''
    Entry point for the tractogram tsf function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tsf_out'], {'type': str, 'help': 'Output tsf file'}],
        [['--orientation', '-o'], {'action': 'store_true', 'default': False, 'help': 'Color based on orientation'}],
        [['--file'], {'type': str, 'help': 'Color based on given file'}]
    ]
    options = setup_parser(tsf_create.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tsf_create(
        options.tractogram_in,
        options.tsf_out,
        options.orientation,
        options.file,
        options.verbose,
        options.force
    )


def tractogram_filter():
    '''
    Entry point for the tractogram filtering function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--minlength', '-minl'], {'type': float, 'help': 'Keep streamlines with length [in mm] >= this value'}],
        [['--maxlength', '-maxl'], {'type': float, 'help': 'Keep streamlines with length [in mm] <= this value'}],
        [['--minweight', '-minw'], {'type': float, 'help': 'Keep streamlines with weight >= this value'}],
        [['--maxweight', '-maxw'], {'type': float, 'help': 'Keep streamlines with weight <= this value'}],
        [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights'}],
        [['--random', '-r'], {'type': float, 'default': 1.0, 'help': '''\
                              Randomly keep the given percentage of streamlines: 0=discard all, 1=keep all. 
                              This filter is applied after all others'''}]
    ]
    options = setup_parser(tract_filter.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tract_filter(
        options.tractogram_in,
        options.tractogram_out,
        options.minlength,
        options.maxlength,
        options.minweight,
        options.maxweight,
        options.weights_in,
        options.weights_out,
        options.random,
        options.verbose,
        options.force
    )


def tractogram_indices():
    '''
    Entry point for the tractogram indices function.
    '''
    # parse the input parameters
    args = [
        [['indices_in'], {'type': str, 'help': 'Indices to recompute'}],
        [['dictionary_kept'], {'type': str, 'help': 'Dictionary of kept streamlines'}],
        [['indices_out'], {'type': str, 'help': 'Output indices file'}]
    ]
    options = setup_parser(recompute_indices.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    recompute_indices(
        options.input_indices,
        options.dictionary_kept,
        options.output_indices,
        verbose=options.verbose
    )


def tractogram_info():
    '''
    Entry point for the tractogram info function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['--lengths', '-l'], {'action': 'store_true', 'help': 'Show stats on streamline lengths'}],
        [['--max_field_length', '-m'], {'type': int, 'help': 'Maximum length allowed for printing a field value'}]
    ]
    options = setup_parser(info.__doc__.split('\n')[0], args)
    
    # call actual function
    info(
        options.tractogram_in,
        options.lengths,
        options.max_field_length
    )


def tractogram_join():
    '''
    Entry point for the tractogram join function.
    '''
    # parse the input parameters
    args = [
        [['tractograms_in'], {'type': str, 'nargs': '*', 'help': 'Input tractograms (2 or more filenames)'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--weights_in'], {'type': str, 'nargs': '*', 'default': [], 'help': 'Input streamline weights (.txt or .npy). NOTE: the order must be the same of the input tractograms'}],
        [['--weights_out'], {'type': str, 'help': 'Output streamline weights (.txt or .npy)'}]
    ]
    options = setup_parser(tract_join.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tract_join( 
        options.tractograms_in,
        options.tractogram_out, 
        options.weights_in,
        options.weights_out,
        options.verbose,
        options.force
    )


def tractogram_join_tsf():
    '''
    Entry point for the tractogram join tsf function.
    '''
    # parse the input parameters
    args = [
        [['tsf_in'], {'type': str, 'nargs': '+', 'help': 'Input tsf files'}],
        [['tsf_out'], {'type': str, 'help': 'Output tsf file'}]
    ]
    options = setup_parser(tract_join.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tsf_join(
        options.tsf_in,
        options.tsf_out,
        options.verbose,
        options.force
    )


def tractogram_lengths():
    '''
    Entry point for the tractogram lengths function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['lengths_out'], {'type': str, 'help': 'Output scalar file (.npy or .txt) that will contain the streamline lengths'}]
    ]
    options = setup_parser(compute_lengths.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    try:
        # call the actual function
        compute_lengths(
            options.tractogram_in,
            options.lengths_out,
            options.verbose,
            options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def tractogram_locate():
    '''
    Entry point for the tractogram find function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_subset_in'], {'type': str, 'help': 'Tractogram containing the subset of streamlines to find'}],
        [['tractogram_in'], {'type': str, 'help': 'Tractogram containing the full set of streamlines in which to search'}],
        [['indices_out'], {'type': str, 'help': 'Output file (.txt or .npy) containing the indices of the subset streamlines in the full tractogram'}]
    ]
    options = setup_parser(get_indices_of_streamlines.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    get_indices_of_streamlines(
        options.tractogram_subset_in,
        options.tractogram_in,
        options.indices_out,
        options.verbose,
        options.force
    )


def tractogram_resample():
    '''
    Entry point for the tractogram resampling function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['n_pts'], {'type': int, 'default': 12, 'metavar': 'N_PTS', 'help': 'Number of points per streamline'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}]
    ]
    options = setup_parser(resample.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    resample(
        options.tractogram_in,
        options.tractogram_out,
        options.n_pts,
        options.verbose,
        options.force,
    )


def tractogram_sample():
    '''
    Entry point for the tractogram sampling function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['image_in'], {'type': str, 'help': 'Input image'}],
        [['file_out'], {'type': str, 'help': 'Path to the file (.txt) where the method saves the sampled values'}],
        [['--mask', '-m'], {'type': str, 'default': None, 'help': 'Optional mask to restrict the sampling voxels'}],
        [['--option'], {'type': str, 'nargs': '?', 'default': 'No_opt', 'choices': ['No_opt', 'mean', 'median', 'min', 'max'], 'help': 'Operation to apply on streamlines (if No_opt: no operation applied'}],
        [['--collapse_pts', '-c'], {'action': 'store_true', 'default': False, 'help': 'Collapse the values of points falling in the same voxel (default : False).'}]
    ]
    options = setup_parser(sample.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    sample(
        options.tractogram_in,
        options.image_in,
        options.file_out,
        options.mask,
        options.option,
        options.collapse_pts,
        options.force,
        options.verbose
    )


def tractogram_sanitize():
    '''
    Entry point for the tractogram sanitization function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['gray_matter'], {'type': str, 'help': 'Path to the gray matter'}],
        [['white_matter'], {'type': str, 'help': 'Path to the white matter'}],
        [['--tractogram_out', '-out'], {'type': str, 'help': 'Output tractogram (if None: "_sanitized" appended to the input filename)'}],
        [['--step', '-s'], {'type': float, 'default': 0.2, 'help': 'Step size [in mm] used to extend or shorten the streamlines'}],
        [['--max_dist', '-d'], {'type': float, 'default': 2, 'help': 'Maximum distance [in mm] used when extending or shortening the streamlines'}],
        [['--save_connecting_tck', '-conn'], {'action': 'store_true', 'default': False, 'help': 'Save also tractogram with only the actual connecting streamlines (if True: "_only_connecting" appended to the output filename)'}]
    ]
    options = setup_parser(sanitize.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    sanitize(
        options.tractogram_in,
        options.gray_matter,
        options.white_matter,
        options.tractogram_out,
        options.step,
        options.max_dist,
        options.save_connecting_tck,
        options.verbose,
        options.force
    )


def tractogram_smooth():
    '''
    Entry point for the tractogram smoothing function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--type', '-t'], {'type': str, 'default': 'centripetal', 'choices': ['uniform', 'chordal', 'centripetal'], 'help': 'Type of spline to use for the smoothing'}],
        [['--segment_len', '-l'], {'type': float, 'default': None, 'help': '''\
                                   Sampling resolution of the final streamline after interpolation. 
                                   NOTE: either "segment_len" or "streamline_pts" must be set, by default "segment_len" is used.
                                         If None and "streamline_pts" is None, "segment_len" is set to 0.5.'''}],
        [['--streamline_pts', '-p'], {'type': int, 'default': None, 'help': '''\
                                      Number of points in each of the final streamlines. 
                                      NOTE: either "streamline_pts" or "segment_len" must be set, by default "segment_len" is used.'''}],
        [['--epsilon', '-e'], {'type': float, 'default': None, 'help': '''\
                               Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline. 
                               NOTE: either "epsilon" or "n_ctrl_pts" must be set, by default "epsilon" is used.
                                     If None and "n_ctrl_pts" is None, "epsilon" is set to 0.3.'''}],
        [['--n_ctrl_pts', '-n'], {'type': int, 'default': None, 'help': '''\
                                  Number of control points used to interpolate the streamlines. 
                                  NOTE: either "epsilon" or "n_ctrl_pts" must be set, by default "epsilon" is used.'''}]
    ]

    options = setup_parser(spline_smoothing_v2.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    spline_smoothing_v2(
        options.tractogram_in,
        options.tractogram_out,
        options.type,
        options.epsilon,
        options.n_ctrl_pts,
        options.segment_len,
        options.streamline_pts,
        options.verbose,
        options.force
    )


def tractogram_sort():
    '''
    Entry point for the tractogram sorting function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['atlas'], {'type': str, 'help': 'Path to the atlas file used to sort the streamlines'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': '''\
                                  Distance used to perform a radial search from each streamline endpoint to locate the nearest node and assign the streamline to the corresponding bundle.
                                  Argument is the maximum radius in mm'''}],
        [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights (.txt or .npy)'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights (.txt or .npy)'}],
        [['--tmp_folder', '-tmp'], {'type': str, 'default': 'tmp', 'metavar': 'TMP_FOLDER', 'help': 'Path to the temporary folder used to store the intermediate files'}],
        [['--keep_temp', '-k'], {'action': 'store_true', 'help': 'Keep temporary folder'}],
        [['--n_threads', '-n'], {'type': int, 'default': 3, 'metavar': 'N_THREADS', 'help': '''\
                                 Number of threads to use.
                                 If None, all the available threads will be used'''}]
    ]
    options = setup_parser(tract_sort.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tract_sort(
        options.tractogram_in,
        options.atlas,
        options.tractogram_out,
        options.atlas_dist,
        options.weights_in,
        options.weights_out,
        options.tmp_folder,
        options.keep_temp,
        options.n_threads,
        options.verbose,
        options.force
    )


def tractogram_split():
    '''
    Entry point for the tractogram splitting function.
    '''
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['assignments_in'], {'type': str, 'help': 'Text file with the streamline assignments'}],
        [['--output_folder', '-out'], {'type': str, 'nargs': '?', 'default': 'bundles', 'help': 'Output folder for the splitted tractograms'}],
        [['--prefix', '-p'], {'type': str, 'default': 'bundle_', 'help': 'Prefix for the output filenames'}],
        [['--regions', '-r'], {'type': str, 'default': None, 'help': '''\
                               Only streamlines connecting the provided region(s) will be extracted.
                               If None, all the bundles (plus the unassigned streamlines) will be extracted.
                               If a single region is provided, all bundles connecting this region with any other will be extracted.
                               If a pair of regions is provided using the format "[r1, r2]", only this specific bundle will be extracted.
                               If list of regions is provided using the format "r1, r2, ...", all the possible bundles connecting one of these regions will be extracted.'''}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': 'Input streamline weights (.txt or .npy)'}],
        [['--max_open', '-m'], {'type': int, 'help': 'Maximum number of files opened at the same time'}]

    ]
    options = setup_parser(split.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    split(
        options.tractogram_in,
        options.assignments_in,
        options.output_folder,
        options.regions,
        options.weights_in,
        options.max_open,
        options.prefix,
        options.verbose,
        options.force
    )
