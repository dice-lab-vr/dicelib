from dicelib.clustering import run_clustering
import dicelib.connectivity
import dicelib.tractogram
from dicelib.ui import setup_logger, setup_parser, get_argparse_info_from_docstring
logger = setup_logger('dice_tractogram')


def assign():
    '''Entry point for the connectivity.assign function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.connectivity.assign.__doc__  )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram']}],
        [['atlas'], {'type': str, 'help': desc['atlas']}],
        [['out_assignments'], {'type': str, 'help': desc['out_assignments']}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'help': desc['atlas_dist']}],
        [['--n_threads', '-n'], {'type': int, 'default': 3, 'metavar': 'N_THREADS', 'help': desc['n_threads']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.connectivity.assign(
            tractogram=options.tractogram,
            atlas=options.atlas,
            out_assignments=options.out_assignments,
            atlas_dist=options.atlas_dist,
            n_threads=options.n_threads,
            verbose=options.verbose,
            force=options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def cluster():
    '''Entry point for the tractogram clustering function'''
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
        [['--save_clust_idx', '-s'], {'action': 'store_true', 'help': 'Save the indices of the cluster to which each input streamline belongs'}],
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
        save_clust_idx=options.save_clust_idx,
        max_open=options.max_open_files
    )


def filter():
    '''Entry point for the tractogram.filter function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.filter.__doc__  )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['out_tractogram'], {'type': str, 'help': desc['out_tractogram_filename']}],
        [['--weights', '-wi'], {'type': str, 'help': desc['weights_filename']}],
        [['--minlength', '-minl'], {'type': float, 'help': desc['minlength']}],
        [['--maxlength', '-maxl'], {'type': float, 'help': desc['maxlength']}],
        [['--minweight', '-minw'], {'type': float, 'help': desc['minweight']}],
        [['--maxweight', '-maxw'], {'type': float, 'help': desc['maxweight']}],
        [['--out_weights', '-wo'], {'type': str, 'help': desc['out_weights_filename']}],
        [['--random', '-r'], {'type': float, 'default': 1.0, 'help': desc['random']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.filter(
            tractogram_filename=options.tractogram,
            out_tractogram_filename=options.out_tractogram,
            weights_filename=options.weights,
            minlength=options.minlength,
            maxlength=options.maxlength,
            minweight=options.minweight,
            maxweight=options.maxweight,
            out_weights_filename=options.out_weights,
            random=options.random,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def recompute_indices():
    '''Entry point for the tractogram.recompute_indices function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.recompute_indices.__doc__  )
    args = [
        [['indices'], {'type': str, 'help': desc['idx_filename']}],
        [['kept'], {'type': str, 'help': desc['kept_filename']}],
        [['out_indices'], {'type': str, 'help': desc['out_idx_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.recompute_indices(
            idx_filename=options.indices,
            kept_filename=options.kept,
            out_idx_filename=options.out_indices,
            verbose=options.verbose,
            force=options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def info():
    '''Entry point for the tractogram.info function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.info.__doc__  )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['--max_field_length', '-m'], {'type': int, 'help': desc['max_field_length']}],
        [['--lengths', '-l'], {'action': 'store_true', 'help': desc['compute_lengths']}]

    ]
    options = setup_parser(summary, args, epilog=notes, add_force=False, add_verbose=True)
    try:
        dicelib.tractogram.info(
            tractogram_filename=options.tractogram,
            max_field_length=options.max_field_length,
            compute_lengths=options.lengths,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def join():
    '''Entry point for the tractogram.join function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.join.__doc__  )
    args = [
        [['tractograms'], {'type': str, 'nargs': '+', 'help': desc['tractograms_filenames']}],
        [['out_tractogram'], {'type': str, 'help': desc['out_tractogram_filename']}],
        [['--weights', '-wi'], {'type': str, 'nargs': '*', 'default': None, 'help': desc['weights_filenames']}],
        [['--out_weights', '-wo'], {'type': str, 'default': None, 'help': desc['out_weights_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.join(
            tractograms_filenames=options.tractograms,
            out_tractogram_filename=options.out_tractogram,
            weights_filenames=options.weights,
            out_weights_filename=options.out_weights,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def compute_lengths():
    '''Entry point for the tractogram.compute_lengths function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.compute_lengths.__doc__ )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['out_scalars'], {'type': str, 'help': desc['out_scalars_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.compute_lengths(
            tractogram_filename=options.tractogram,
            out_scalars_filename=options.out_scalars,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def locate():
    '''Entry point for the tractogram.get_indices_of_streamlines function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.get_indices_of_streamlines.__doc__ )
    args = [
        [['tractogram_needle'], {'type': str, 'help': desc['needle_filename']}],
        [['tractogram_haystack'], {'type': str, 'help': desc['haystack_filename']}],
        [['out_idx'], {'type': str, 'help': desc['out_idx_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.get_indices_of_streamlines(
            needle_filename=options.tractogram_needle,
            haystack_filename=options.tractogram_haystack,
            out_idx_filename=options.out_idx,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def resample():
    '''Entry point for the tractogram.resample function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.resample.__doc__ )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['n_pts'], {'type': int, 'help': desc['n_pts']}],
        [['out_tractogram'], {'type': str, 'help': desc['out_tractogram_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.resample(
            tractogram_filename=options.tractogram,
            out_tractogram_filename=options.out_tractogram,
            n_pts=options.n_pts,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def sample():
    '''Entry point for the tractogram.sample function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.sample.__doc__ )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['image'], {'type': str, 'help': desc['image_filename']}],
        [['out_scalars'], {'type': str, 'help': desc['out_scalars_filename']}],
        [['--mask', '-m'], {'type': str, 'default': None, 'help': desc['mask_filename']}],
        [['--option', '-o'], {'type': str, 'nargs': '?', 'default': 'no_opt', 'choices': ['no_opt', 'mean', 'median', 'min', 'max'], 'help': desc['option']}],
        [['--collapse', '-c'], {'action': 'store_true', 'default': False, 'help': desc['collapse']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.sample(
            tractogram_filename=options.tractogram,
            image_filename=options.image,
            out_scalars_filename=options.out_scalars,
            mask_filename=options.mask,
            option=options.option,
            collapse=options.collapse,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def sanitize():
    '''Entry point for the tractogram sanitization function'''
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['gray_matter'], {'type': str, 'help': 'Path to the gray matter'}],
        [['white_matter'], {'type': str, 'help': 'Path to the white matter'}],
        [['--tractogram_out', '-out'], {'type': str, 'help': 'Output tractogram (if None: "_sanitized" appended to the input filename)'}],
        [['--step', '-s'], {'type': float, 'default': 0.2, 'help': 'Step size [in mm] used to extend or shorten the streamlines'}],
        [['--max_dist', '-d'], {'type': float, 'default': 2, 'help': 'Maximum distance [in mm] used when extending or shortening the streamlines'}],
        [['--save_connecting_tck', '-conn'], {'action': 'store_true', 'default': False, 'help': 'Save also tractogram with only the actual connecting streamlines (if True: "_only_connecting" appended to the output filename)'}]
    ]
    options = setup_parser(dicelib.tractogram.sanitize.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    dicelib.tractogram.sanitize(
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


def shuffle():
    '''Entry point for the tractogram shuffling function'''
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--n_tmp_groups', '-g'], {'type': int, 'default': 100, 'help': 'Number of temporary groups used to shuffle the streamlines'}],
        [['--seed', '-s'], {'type': int, 'default': None, 'help': 'Seed used for the random shuffling'}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': 'Input streamline weights (.txt or .npy)'}],
        [['--weights_out', '-o'], {'type': str, 'default': None, 'help': 'Output streamline weights (.txt or .npy)'}],
        [['--tmp_folder', '-tmp'], {'type': str, 'default': 'tmp_shuffle', 'metavar': 'TMP_FOLDER', 'help': 'Path to the temporary folder used to store the intermediate files'}],
        [['--keep_tmp', '-k'], {'action': 'store_true', 'help': 'Keep temporary folder'}]
    ]
    options = setup_parser(dicelib.tractogram.shuffle.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    dicelib.tractogram.shuffle(
        options.tractogram_in,
        options.tractogram_out,
        options.n_tmp_groups,
        options.seed,
        options.weights_in,
        options.weights_out,
        options.tmp_folder,
        options.keep_tmp,
        options.verbose,
        options.force
    )


def smooth():
    '''Entry point for the tractogram smoothing function'''
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--type', '-t'], {'type': str, 'default': 'centripetal', 'choices': ['uniform', 'chordal', 'centripetal'], 'help': 'Type of spline to use for the smoothing'}],
        [['--epsilon', '-e'], {'type': float, 'default': None, 'help': '''\
                               Distance threshold used by Ramer-Douglas-Peucker algorithm to choose the control points of the spline.
                               NOTE: either "epsilon" or "n_ctrl_pts" must be set, by default "epsilon" is used.
                                     If None and "n_ctrl_pts" is None, "epsilon" is set to 0.3.'''}],
        [['--n_ctrl_pts', '-n'], {'type': int, 'default': None, 'help': '''\
                                  Number of control points used to interpolate the streamlines.
                                  NOTE: either "epsilon" or "n_ctrl_pts" must be set, by default "epsilon" is used.'''}],
        [['--do_resample', '-r'], {'action': 'store_true', 'default': False, 'help': '''\
                                   If True, the final streamlines are resampled to have a constant segment length (see "segment_len" and "streamline_pts" parameters).
                                   If False, the point of the final streamlines are more dense where the curvature is high.'''}],
        [['--segment_len', '-l'], {'type': float, 'default': None, 'help': '''\
                                   Sampling resolution of the final streamline after interpolation.
                                   NOTE: if 'do_resample' is True, either "segment_len" or "streamline_pts" must be set, by default "segment_len" is used.
                                   If None and "streamline_pts" is None, "segment_len" is set to 0.5.'''}],
        [['--streamline_pts', '-p'], {'type': int, 'default': None, 'help': '''\
                                      Number of points in each of the final streamlines.
                                      NOTE: if 'do_resample' is True, either "streamline_pts" or "segment_len" must be set, by default "segment_len" is used.'''}],
        [['--n_pts_eval', '-n_ev'], {'type': int, 'default': None, 'help': '''\
                                     Number of points in which the spline is evaluated.
                                     If None, the number of points is computed using "segment_len_eval"'''}],
        [['--segment_len_eval', '-l_ev'], {'type': float, 'default': None, 'help': '''\
                                           Segment length used to compute the number of points in which the spline is evaluated; computed as the length of the reduced streamline divided by "segment_len_eval".
                                           If None and "n_pts_eval" is None, "segment_len_eval" is set to 0.5'''}]
    ]
    options = setup_parser(dicelib.tractogram.spline_smoothing_v2.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    dicelib.tractogram.spline_smoothing_v2(
        options.tractogram_in,
        options.tractogram_out,
        options.type,
        options.epsilon,
        options.n_ctrl_pts,
        options.n_pts_eval,
        options.segment_len_eval,
        options.do_resample,
        options.segment_len,
        options.streamline_pts,
        options.verbose,
        options.force
    )


def sort():
    '''Entry point for the tractogram sorting function'''
    args = [
        [['tractogram_in'], {'type': str, 'help': 'Input tractogram'}],
        [['atlas'], {'type': str, 'help': 'Path to the atlas file used to sort the streamlines'}],
        [['tractogram_out'], {'type': str, 'help': 'Output tractogram'}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'metavar': 'ATLAS_DIST', 'help': '''\
                                  Distance used to perform a radial search from each streamline endpoint to locate the nearest node and assign the streamline to the corresponding bundle.
                                  Argument is the maximum radius in mm'''}],
        [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights (.txt or .npy)'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights (.txt or .npy)'}],
        [['--tmp_folder', '-tmp'], {'type': str, 'default': 'tmp_sort', 'metavar': 'TMP_FOLDER', 'help': 'Path to the temporary folder used to store the intermediate files'}],
        [['--keep_temp', '-k'], {'action': 'store_true', 'help': 'Keep temporary folder'}],
        [['--n_threads', '-n'], {'type': int, 'default': 3, 'metavar': 'N_THREADS', 'help': '''\
                                 Number of threads to use.
                                 If None, all the available threads will be used'''}]
    ]
    options = setup_parser(dicelib.tractogram.sort.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    dicelib.tractogram.sort(
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


def split():
    '''Entry point for the tractogram.split function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.split.__doc__  )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['assignments'], {'type': str, 'help': desc['assignments_filename']}],
        [['--out_folder', '-out'], {'type': str, 'nargs': '?', 'default': 'bundles', 'help': desc['out_folder']}],
        [['--prefix', '-p'], {'type': str, 'default': 'bundle_', 'help': desc['prefix']}],
        [['--regions', '-r'], {'type': str, 'default': None, 'help': desc['regions']}],
        [['--weights', '-w'], {'type': str, 'default': None, 'help': desc['weights_filename']}],
        [['--max_open', '-m'], {'type': int, 'default': None, 'help': desc['max_open']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.split(
            tractogram_filename=options.tractogram,
            assignments_filename=options.assignments,
            out_folder=options.out_folder,
            prefix=options.prefix,
            regions=options.regions,
            weights_filename=options.weights,
            max_open=options.max_open,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def compute_coherence():
    '''Entry point for the tractogram.compute_coherence function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.compute_coherence.__doc__ )
    args = [
        [['tractogram'], {'help': desc['tractogram_filename']}],
        [['sph_func'], {'help': desc['sph_func_filename']}],
        [['out_weights'], {'help': desc['out_weights_filename']}],
        [['--metric', '-m'], {'choices': ['min','mean','max'], 'default': 'min', 'help': desc['metric']}],
        [['--normalize', '-n'], {'action': 'store_true', 'help': desc['normalize']}],
        [['--trim', '-t'], {'type': float, 'default': 0.05, 'help': desc['trim']}],
        [['--shift', '-s'], {'type': float, 'default': 0.5, 'help': desc['shift']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.compute_coherence(
            tractogram_filename=options.tractogram,
            sph_func_filename=options.sph_func,
            out_weights_filename=options.out_weights,
            metric=options.metric,
            normalize=options.normalize,
            trim=options.trim,
            shift=options.shift,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def compute_tdi():
    '''Entry point for the tractogram.compute_tdi function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.compute_tdi.__doc__  )
    args = [
        [['tractogram'], {'help': desc['tractogram_filename']}],
        [['ref_image'], {'help': desc['ref_image_filename']}],
        [['out_map'], {'help': desc['out_map_filename']}],
        [['--shift', '-s'], {'type': float, 'default': 0.5, 'help': desc['shift']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.compute_tdi(
            tractogram_filename=options.tractogram,
            ref_image_filename=options.ref_image,
            out_map_filename=options.out_map,
            shift=options.shift,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def tsf_create():
    '''Entry point for the tractogram.tsf_create function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.tsf_create.__doc__ )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram']}],
        [['scalars'], {'type': str, 'help': desc['scalars']}],
        [['out_tsf'], {'type': str, 'help': desc['out_tsf']}],
        [['--check_orientation', '-check'], {'action': 'store_true', 'default': False, 'help': desc['check_orientation']}],
        [['--out_tractogram'], {'type': str, 'default': None, 'help': desc['out_tractogram']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.tsf_create(
            tractogram=options.tractogram,
            scalars=options.scalars,
            out_tsf=options.out_tsf,
            check_orientation=options.check_orientation,
            out_tractogram=options.out_tractogram,
            verbose=options.verbose,
            force=options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def tsf_join():
    '''Entry point for the tractogram.tsf_join function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tractogram.tsf_join.__doc__ )
    args = [
        [['tsf_in'], {'type': str, 'nargs': '+', 'help': 'Input tsf files'}],
        [['tsf_out'], {'type': str, 'help': 'Output tsf file'}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tractogram.tsf_join(
            options.tsf_in,
            options.tsf_out,
            options.verbose,
            options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


# def compress():
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


# def convert():
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