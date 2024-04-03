from dicelib.clustering import run_clustering
from dicelib.connectivity import assign
from dicelib.tractogram import compute_lengths, filter as tract_filter, info, join as tract_join, LazyTractogram, recompute_indices, resample, sample, sanitize, spline_smoothing_v2, split
from dicelib.tsf import Tsf
from dicelib.ui import __logger__ as logger, ProgressBar, set_verbose, setup_parser

import os
from time import time

import numpy as np


def create_color_scalar_file(streamline, num_streamlines):
        """
        Create a scalar file for each streamline in order to color them.
        Parameters
        ----------
        streamlines: list
            List of streamlines.
        Returns
        -------
        scalar_file: str
            Path to scalar file.
        """
        scalar_list = list()
        n_pts_list = list()
        for i in range(num_streamlines):
            # pt_list = list()
            streamline.read_streamline()
            n_pts_list.append(streamline.n_pts)
            for j in range(streamline.n_pts):
                scalar_list.extend([float(j)])
            # scalar_list.append(pt_list)
        return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)


def color_by_scalar_file(streamline, values, num_streamlines):
    """
    Color streamlines based on sections.
    Parameters
    ----------
    streamlines: array
        Array of streamlines.
    values: list
        List of scalars.
    Returns
    -------
    array
        Array mapping scalar values to each vertex of each streamline.
    array
        Array containing the number of points of each input streamline.
    """
    scalar_list = []
    n_pts_list = []
    for i in range(num_streamlines):
        streamline.read_streamline()
        n_pts_list.append(streamline.n_pts)
        streamline_points = np.arange(streamline.n_pts)
        resample = np.linspace(0, streamline.n_pts, len(values), endpoint=True, dtype=np.int32)
        streamline_points = np.interp(streamline_points, resample, values)
        scalar_list.extend(streamline_points)
    return np.array(scalar_list, dtype=np.float32), np.array(n_pts_list, dtype=np.int32)


def tractogram_assign():
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['atlas'], {'type': str, 'help': 'Atlas used to compute streamlines assignments'}],
        [['assignments_out'], {'type': str, 'help': 'Output assignments file (.txt or .npy)'}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': 'Distance [in mm] used to assign streamlines to the atlas\' nodes'}]
    ]
    options = setup_parser(assign.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    set_verbose(options.verbose)

    # check if tractogram exists
    if not os.path.exists(options.tractogram_in):
        logger.error('Tractogram does not exist')

    # check if path to save assignments is relative or absolute and create if necessary
    if options.assignments_out:
        if not os.path.isabs(options.assignments_out):
            options.assignments_out = os.path.join(os.getcwd(), options.assignments_out)
        if not os.path.isdir(os.path.dirname(options.assignments_out)):
            os.makedirs(os.path.dirname(options.assignments_out))

    out_assignment_ext = os.path.splitext(options.assignments_out)[1]
    if out_assignment_ext not in ['.txt', '.npy']:
        logger.error('Invalid extension for the output scalar file')
    elif os.path.isfile(options.assignments_out) and not options.force:
        logger.error('Output scalar file already exists, use -f to overwrite')


    # check if atlas exists
    if not os.path.exists(options.atlas):
        logger.error('Atlas does not exist')

    assign(options.tractogram_in,
           options.assignments_out,
           options.atlas,
           options.atlas_dist)


def tractogram_cluster():
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['clust_thr'], {'type': float, 'help': 'Distance threshold [in mm] used to cluster the streamlines'}],
        [['tractogram_out'], {'type': str, 'default': None, 'help': 'Output clustered tractogram'}],
        [['--metric', '-m'], {'type': str, 'default': 'mean', 'metavar': 'METRIC', 'help': 'Metric used to cluster the streamlines. Options: \'mean\', \'max\''}],
        [['--n_pts', '-n'], {'type': int, 'default': 12, 'metavar': 'N_PTS', 'help': 'Number of points for the resampling of a streamline'}],
        [['--atlas', '-a'], {'type': str, 'metavar': 'ATLAS_FILE', 'help': 'Path to the atlas file used to split the streamlines into bundles. If provided, parallel clustering will be performed'}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': 'Distance [in mm] used to assign streamlines to the atlas\' nodes for hierarchical clustering'}],
        [['--tmp_folder', '-tmp'], {'type': str, 'default': 'tmp', 'metavar': 'TMP_FOLDER', 'help': 'Path to the temporary folder used to store the intermediate files'}],
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
        clust_thr=options.clust_thr,
        metric=options.metric,
        n_pts=options.n_pts,
        n_threads=options.n_threads,
        force=options.force,
        verbose=options.verbose,
        keep_temp_files=options.keep_temp
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


def tractogram_filter():
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
        [['--random', '-r'], {'type': float, 'default': 1.0, 'help': 'Randomly discard streamlines: 0=discard all, 1=keep all'}]
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
    # parse the input parameters
    args = [
        [['input_indices'], {'type': str, 'help': 'Indices to recompute'}],
        [['dictionary_kept'], {'type': str, 'help': 'Dictionary of kept streamlines'}],
        [['output_indices'], {'type': str, 'help': 'Output indices file'}]
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


def tractogram_lengths():
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


def tractogram_resample():
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--n_pts', '-n'], {'type': int, 'default': 12, 'metavar': 'N_PTS', 'help': 'Number of points per streamline'}]
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
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['image_in'], {'type': str, 'help': 'Input image'}],
        [['file_out'], {'type': str, 'help': 'File for the output'}],
        [['--mask', '-m'], {'type': str, 'default': None, 'help': 'Optional mask to restrict the sampling voxels'}],
        [['--space'], {'type': str, 'nargs': '?', 'default': 'rasmm', 'choices': ['voxmm', 'rasmm', 'vox'], 'help': 'Current reference space of streamlines (rasmm, voxmm, vox)'}],
        [['--option'], {'type': str, 'nargs': '?', 'default': 'No_opt', 'choices': ['No_opt', 'mean', 'median', 'min', 'max'], 'help': 'Operation to apply on streamlines (if No_opt: no operation applied'}]
    ]
    options = setup_parser(sample.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    sample(
        options.tractogram_in,
        options.image_in,
        options.file_out,
        options.mask,
        options.space,
        options.option,
        options.force,
        options.verbose
    )


def tractogram_sanitize():
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['gray_matter'], {'type': str, 'help': 'Gray matter'}],
        [['white_matter'], {'type': str, 'help': 'White matter'}],
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
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--type', '-t'], {'type': str, 'default': 'centripetal', 'choices': ['uniform', 'chordal', 'centripetal'], 'help': 'Type of spline to use for the smoothing'}],
        [['--epsilon', '-e'], {'type': float, 'default': 0.3, 'help': 'Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline'}],
        [['--segment_len', '-l'], {'type': float, 'default': None, 'help': 'Sampling resolution of the final streamline after interpolation. NOTE: either "segment_len" or "streamline_pts" must be set'}],
        [['--streamline_pts', '-p'], {'type': int, 'default': None, 'help': 'Number of points in each of the final streamlines. NOTE: either "streamline_pts" or "segment_len" must be set.'}]
    ]

    options = setup_parser(spline_smoothing_v2.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    spline_smoothing_v2(
        options.tractogram_in,
        options.tractogram_out,
        options.type,
        options.epsilon,
        options.segment_len,
        options.streamline_pts,
        options.verbose,
        options.force
    )


def tractogram_split():
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['assignments_in'], {'type': str, 'help': 'Text file with the streamline assignments'}],
        [['--output_folder', '-out'], {'type': str, 'nargs': '?', 'default': 'bundles', 'help': 'Output folder for the splitted tractograms'}],
        [['--regions', '-r'], {'type': str, 'default': None, 'help': '''\
                               Streamline connecting the provided region(s) will be extracted.
                               If None, all the bundles (plus the unassigned streamlines) will be extracted.
                               If a single region is provided, all bundles connecting this region with any other will be extracted.
                               If a pair of regions is provided using the format "[r1, r2]", only this specific bundle will be extracted.
                               If list of regions is provided using the format "r1, r2, ...", all the possible bundles connecting one of these regions will be extracted.'''}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': 'Input streamline weights (.txt or .npy)'}],
        [['--max_open', '-m'], {'type': int, 'help': 'Maximum number of files opened at the same time'}],
        [['--prefix', '-p'], {'type': str, 'default': 'bundle_', 'help': 'Prefix for the output filenames'}]

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


def tractogram_tsf():
    # parse the input parameters
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tsf_out'], {'type': str, 'help': 'Output tsf filename'}],
        [['--orientation', '-o'], {'action': 'store_true', 'default': False, 'help': 'Color based on orientation'}],
        [['--file', '-f'], {'type': str, 'help': 'Color based on given file'}]
    ]
    options = setup_parser('Create a tsf file for each streamline in order to color them.', args, add_force=True)

    # check if path to input and output files are valid
    if not os.path.isfile(options.tractogram_in):
        logger.error(f'Input tractogram file not found: {options.tractogram_in}')
    if os.path.isfile(options.output_tsf) and not options.force:
        logger.error('Output file already exists. Use -f to overwrite.')
    if not options.orientation and not options.file:
        logger.error("Please specify a color option")
    if options.file:
        if not os.path.isfile(options.file):
            logger.error(f"Input file not found: {options.file}")

    streamline = LazyTractogram(options.tractogram_in, mode='r')
    num_streamlines = streamline.header['count']

    if options.orientation:
        scalar_arr, n_pts_list = create_color_scalar_file(streamline, int(num_streamlines))
    elif options.file:
        values = np.loadtxt(options.file)
        scalar_arr, n_pts_list = color_by_scalar_file(streamline, values, int(num_streamlines))
    else:
        raise ValueError("Please specify a color option")

    # check if output file exists
    if os.path.isfile(options.output_tsf) and not options.force:
        logger.error('Output file already exists. Use -f to overwrite.')

    tsf = Tsf(options.tsf_out, 'w', header=streamline.header)
    tsf.write_scalar(scalar_arr, n_pts_list)
