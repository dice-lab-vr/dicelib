from dicelib.clustering import run_clustering
from dicelib.connectivity import assign
from dicelib.lazytractogram import LazyTractogram
from dicelib.tractogram import compute_lengths, filter as t_filter, info, join as t_join, recompute_indices, resample, sample, sanitize, spline_smoothing, split
from dicelib.tsf import Tsf
from dicelib.ui import ColoredArgParser, ERROR, INFO, ProgressBar, set_verbose, WARNING

from concurrent.futures import ThreadPoolExecutor
from dipy.io.stateful_tractogram import set_sft_logger_level
from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np
from os import getcwd, makedirs, remove
from os.path import dirname, exists, isabs, isfile, isdir, join as p_join, splitext 
from time import time

def compute_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def setup_parser(description: str, args: list):
    parser = ColoredArgParser(description=description)
    for arg in args:
        parser.add_argument(*arg[0], **arg[1])
    return parser.parse_args()

def tractogram_assign():
    args = [
        [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['atlas'], {'type': str, 'help': 'Atlas used to compute streamlines assignments'}],
        [['--conn_threshold', '-t'], {'type': float, 'default': 2, 'metavar': 'CONN_THR', 'help': 'Threshold [in mm]'}],
        [['--save_assignments'], {'type': str, 'metavar': 'ASSIGNMENTS_FILE', 'help': 'Save the cluster assignments to file'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'metavar': 'VERBOSE_LEVEL', 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(assign.__doc__.split('\n')[0], args)

    set_verbose(options.verbose)

    out_assignment_ext = splitext(options.save_assignments)[1]
    if out_assignment_ext not in ['.txt', '.npy']:
        ERROR('Invalid extension for the output scalar file')
    elif isfile(options.save_assignments) and not options.force:
        ERROR('Output scalar file already exists, use -f to overwrite')

    # check if path to save assignments exists and create it if not
    if not exists(dirname(options.save_assignments)):
        makedirs(dirname(options.save_assignments))

    # check if atlas exists
    if not exists(options.atlas):
        ERROR('Atlas does not exist')

    num_streamlines = int(LazyTractogram(options.tractogram, mode='r').header["count"])
    INFO(f"Computing assignments for {num_streamlines} streamlines")

    if num_streamlines > 3:
        MAX_THREAD = 3
    else:
        MAX_THREAD = 1

    chunk_size = int(num_streamlines / MAX_THREAD)
    chunk_groups = [e for e in compute_chunks(np.arange(num_streamlines), chunk_size)]
    chunks_asgn = []
    
    pbar_array = np.zeros(MAX_THREAD, dtype=np.int32)
    t0 = time()
    with ProgressBar(multithread_progress=pbar_array, total=num_streamlines, disable=(options.verbose in [0, 1, 3]), hide_on_exit=True) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_THREAD) as executor:
            future = [
                executor.submit(
                    assign,
                    options.tractogram,
                    pbar_array,
                    i,
                    start_chunk=int(chunk_groups[i][0]),
                    end_chunk=int(chunk_groups[i][len(chunk_groups[i]) - 1] + 1),
                    gm_map_file=options.atlas,
                    threshold=options.conn_threshold)
                    for i in range(len(chunk_groups)
                )
            ]
            chunks_asgn = [f.result() for f in future]
            chunks_asgn = [c for f in chunks_asgn for c in f]

    t1 = time()
    INFO(f"Time taken to compute assignments: {np.round((t1-t0),2)} seconds")

    if out_assignment_ext == '.txt':
        with open(options.save_assignments, "w") as text_file:
            for reg in chunks_asgn:
                print('%d %d' % (int(reg[0]), int(reg[1])), file=text_file)
    else:
        np.save(options.save_assignments, chunks_asgn, allow_pickle=False)

def tractogram_cluster():
    # parse the input parameters
    args = [
        [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['clust_thr'], {'type': float, 'help': 'Distance threshold [in mm] used to cluster the streamlines'}],
        [['file_name_out'], {'type': str, 'default': None, 'help': 'Output clustered tractogram'}],
        [['--atlas', '-a'], {'type': str, 'metavar': 'ATLAS_FILE', 'help': 'Atlas used to compute streamlines connectivity'}],
        [['--conn_thr', '-t'], {'type': float, 'default': 2, 'metavar': 'CONN_THR', 'help': 'Threshold [in mm]'}],
        [['--metric'], {'type': str, 'default': 'mean', 'metavar': 'METRIC', 'help': 'Metric used to cluster the streamlines. Options: "mean", "max (default: "mean").'}],
        [['--n_pts'], {'type': int, 'default': 10, 'metavar': 'N_PTS', 'help': 'Number of points for the resampling of a streamline'}],
        [['--save_assignments'], {'type': str, 'metavar': 'ASSIGNMENTS_FILE', 'help': 'Save the cluster assignments to file'}],
        [['--output_folder', '-out'], {'type': str, 'metavar': 'OUT_FOLDER', 'help': 'Folder where to save the split clusters'}],
        [['--n_threads'], {'type': int, 'metavar': 'N_THREADS', 'help': 'Number of threads to use to perform clustering'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'metavar': 'VERBOSE_LEVEL', 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}],
        [['--delete_temp', '-d'], {'default': True, 'action': 'store_true', 'help': 'Delete temporary files'}]
    ]
    options = setup_parser(run_clustering.__doc__.split('\n')[0], args)

    # file_name_in: str, output_folder: str=None, file_name_out: str=None, atlas: str=None, conn_thr: float=2.0,
    #                 clust_thr: float=2.0, metric: str="mean", n_pts: int=10, save_assignments: str=None, temp_idx: str=None,
    #                 n_threads: int=None, force: bool=False, verbose: int=1):

    # check the input parameters
    # check if path to input and output files are valid
    if not isfile(options.tractogram):
        ERROR("Input file does not exist: %s" % options.tractogram)

    if options.file_name_out is not None:
        out_ext = splitext(options.file_name_out)[1]
        if out_ext not in ['.trk', '.tck']:
            ERROR('Invalid extension for the output tractogram')
        elif isfile(options.file_name_out) and not options.force:
            ERROR('Output tractogram already exists, use -f to overwrite')

    # check if path to save assignments exists and create it if not
    if options.save_assignments is not None:
        out_assignment_ext = splitext(options.save_assignments)[1]
        if out_assignment_ext not in ['.txt', '.npy']:
            ERROR('Invalid extension for the output scalar file')
        elif isfile(options.save_assignments) and not options.force:
            ERROR('Output scalar file already exists, use -f to overwrite')

        if not exists(dirname(options.save_assignments)):
            makedirs(dirname(options.save_assignments))

    # check if atlas exists
    if options.atlas is not None:
        if not exists(options.atlas):
            ERROR('Atlas does not exist')

    # check if output folder exists
    if options.output_folder is not None:
        if not exists(options.output_folder):
            makedirs(options.output_folder)

    # check if number of threads is valid
    if options.n_threads is not None:
        if options.n_threads < 1:
            ERROR('Number of threads must be at least 1')

    # check if connectivity threshold is valid
    if options.conn_thr is not None:
        if options.conn_thr < 0:
            ERROR('Connectivity threshold must be positive')

    # check if clustering threshold is valid
    if options.clust_thr is not None:
        if options.clust_thr < 0:
            ERROR('Clustering threshold must be positive')

    run_clustering(
        file_name_in=options.tractogram,
        output_folder=options.output_folder,
        file_name_out=options.file_name_out,
        atlas=options.atlas,
        conn_thr=options.conn_thr,
        clust_thr=options.clust_thr,
        metric=options.metric,
        n_pts=options.n_pts,
        save_assignments=options.save_assignments,
        n_threads=options.n_threads,
        force=options.force,
        verbose=options.verbose,
        delete_temp_files=options.delete_temp
    )

def tractogram_compress():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_tractogram'], {'type': str, 'help': 'Output tractogram'}],
        [['--minlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] >= this value'}],
        [['--maxlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] <= this value'}],
        [['--minweight'], {'type': float, 'help': 'Keep streamlines with weight >= this value'}],
        [['--maxweight'], {'type': float, 'help': 'Keep streamlines with weight <= this value'}],
        [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'metavar': 'VERBOSE_LEVEL', 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser('Not implemented', args)

    WARNING('This function is not implemented yet')

def tractogram_convert():
    set_sft_logger_level("CRITICAL")
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_tractogram'], {'type': str, 'help': 'Output tractogram'}],
        [['--reference', '-r'], {'type': str, 'help': 'Space attributes used as reference for the input tractogram'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}]
    ]
    options = setup_parser("Tractogram conversion from and to '.tck', '.trk', '.fib', '.vtk' and 'dpy'. All the extensions except '.trk, need a NIFTI file as reference", args)

    if not isfile(options.input_tractogram):
        ERROR("No such file {}".format(options.input_tractogram))
    if isfile(options.output_tractogram) and not options.force:
        ERROR("Output tractogram already exists, use -f to overwrite")
    if options.reference is not None:
        if not isfile(options.reference):
            ERROR("No such file {}".format(options.reference))

    if not options.input_tractogram.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        ERROR("Invalid input tractogram format")
    elif not options.output_tractogram.endswith(('.tck', '.trk', '.fib', '.vtk', 'dpy')):
        ERROR("Invalid input tractogram format")
    elif options.reference is not None and not options.reference.endswith(('.nii', 'nii.gz')):
        ERROR("Invalid reference format")

    if options.input_tractogram.endswith('.tck') and options.reference is None:
        ERROR("Reference is required if the input format is '.tck'")

    try:
        sft_in = load_tractogram(
            options.input_tractogram,
            reference=options.reference if options.reference else "same"
        )
    except Exception:
        raise ValueError("Error loading input tractogram")
    
    try:
        save_tractogram(sft_in, options.output_tractogram)
    except (OSError, TypeError) as e:
        ERROR(f"Output not valid: {e}")

def tractogram_filter():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_tractogram'], {'type': str, 'help': 'Output tractogram'}],
        [['--minlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] >= this value'}],
        [['--maxlength'], {'type': float, 'help': 'Keep streamlines with length [in mm] <= this value'}],
        [['--minweight'], {'type': float, 'help': 'Keep streamlines with weight >= this value'}],
        [['--maxweight'], {'type': float, 'help': 'Keep streamlines with weight <= this value'}],
        [['--weights_in'], {'type': str, 'help': 'Text file with the input streamline weights'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights'}],
        [['--random', '-r'], {'type': float, 'default': 1.0, 'help': 'Randomly discard streamlines: 0=discard all, 1=keep all'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(t_filter.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.input_tractogram):
        ERROR(f"Input tractogram file not found: {options.input_tractogram}")
    if isfile(options.output_tractogram) and not options.force:
        ERROR(f"Output tractogram file already exists: {options.output_tractogram}")
    # check if the output tractogram file has the correct extension
    output_tractogram_ext = splitext(options.output_tractogram)[1]
    if output_tractogram_ext not in ['.trk', '.tck']:
        ERROR("Invalid extension for the output tractogram file")

    # check if output tractogram file has absolute path and if not, add the
    # current working directory
    if not isabs(options.output_tractogram):
        options.output_tractogram = p_join(getcwd(), options.output_tractogram)

    # check if the input weights file is valid
    if options.weights_in:
        if not isfile(options.weights_in):
            ERROR(f"Input weights file not found: {options.weights_in}")
        if options.weights_out and isfile(options.weights_out) and not options.force:
            ERROR(f"Output weights file already exists: {options.weights_out}")

    # check if the input weights file has absolute path and if not, add the
    # current working directory
    if options.weights_in and not isabs(options.weights_in):
        options.weights_in = p_join(getcwd(), options.weights_in)

    # check if the output weights file has absolute path and if not, add the
    # current working directory
    if options.weights_out and not isabs(options.weights_out):
        options.weights_out = p_join(getcwd(), options.weights_out)

    # call actual function
    t_filter(
        options.input_tractogram,
        options.output_tractogram,
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
        [['indices'], {'type': str, 'help': 'Indices to recompute'}],
        [['dictionary_kept'], {'type': str, 'help': 'Dictionary of kept streamlines'}],
        [['--output', '-o'], {'type': str, 'dest': 'indices_recomputed', 'help': 'Output indices file'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(recompute_indices.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.indices):
        ERROR(f"Input indices file not found: {options.indices}")
    if not isfile(options.dictionary_kept):
        ERROR(f"Input dictionary file not found: {options.dictionary_kept}")
    if isfile(options.indices_recomputed) and not options.force:
        ERROR(
            f"Output indices file already exists: {options.indices_recomputed}")

    # call actual function
    new_indices = recompute_indices(
        options.indices,
        options.dictionary_kept,
        verbose=options.verbose
    )

    # save new indices
    if options.indices_recomputed:
        np.savetxt(options.indices_recomputed, new_indices, fmt='%d')

def tractogram_info():
    # parse the input parameters
    args = [
        [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['--lengths', '-l'], {'action': 'store_true', 'help': 'Show stats on streamline lengths'}],
        [['--max_field_length', '-m'], {'type': int, 'help': 'Maximum length allowed for printing a field value'}]
    ]
    options = setup_parser(info.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.tractogram):
        ERROR(f'Input tractogram file not found: {options.tractogram}')
    
    # call actual function
    info(
        options.tractogram,
        options.lengths,
        options.max_field_length
    )

def tractogram_join():
    # parse the input parameters
    args = [
        [['input_tractograms'], {'type': str, 'nargs': '*', 'help': 'Input tractograms'}],
        [['output_tractogram'], {'type': str, 'help': 'Output tractogram'}],
        [['--input_weights'], {'type': str, 'nargs': '*', 'default': [], 'help': 'Text files with the input streamline weights. NOTE: the order must be the same of the input tractograms'}],
        [['--weights_out'], {'type': str, 'help': 'Text file for the output streamline weights'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(t_join.__doc__.split('\n')[0], args)

    pwd = getcwd()

    # check if path to output file is valid
    if isfile(options.output_tractogram) and not options.force:
        ERROR(f"Output tractogram file already exists: {options.output_tractogram}")
    # check if the output tractogram file has the correct extension
    output_tractogram_ext = splitext(options.output_tractogram)[1]
    if output_tractogram_ext != '.tck':
        ERROR('Invalid extension for the output tractogram file, must be ".tck"')
    # check if output tractogram file has absolute path and if not, add the
    # current working directory
    if not isabs(options.output_tractogram):
        options.output_tractogram = p_join(pwd, options.output_tractogram)

    # check if enough tractograms are given in input
    if len(options.input_tractograms) < 2:
        ERROR("Too few tractograms provided in input, only 2 or more are allowed")
    # check if path to input files is valid
    for f in options.input_tractograms:
        if not isfile( f ):
            ERROR(f"Input tractogram file not found: {f}")
        if splitext(f)[1] != '.tck':
            ERROR(f'Invalid extension for the input tractogram {f}, must be ".tck"')

    if options.input_weights:
        for i,w in enumerate(options.input_weights):
            # check if the input weights file is valid
            if not isfile(w):
                ERROR(f"Input weights file not found: {w}")
            # check if the input weights file has absolute path and if not, add the current working directory
            if not isabs(w):
                options.input_weights[i] = p_join(pwd, w)
        if options.weights_out and isfile(options.weights_out) and not options.force:
            # check if the output weights file is valid
            ERROR(f"Output weights file already exists: {options.weights_out}")
        # check if the output weights file has absolute path and if not, add the current working directory
        if options.weights_out and not isabs(options.weights_out):
            options.weights_out = p_join(pwd, options.weights_out)

    # call actual function
    t_join( 
        options.input_tractograms,
        options.output_tractogram, 
        options.input_weights,
        options.weights_out,
        options.verbose,
        options.force
    )

def tractogram_lengths():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_scalar_file'], {'type': str, 'help': 'Output scalar file (.npy or .txt) that will contain the streamline lengths'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(compute_lengths.__doc__.split('\n')[0], args)

    # check for errors
    output_scalar_file_ext = splitext(options.output_scalar_file)[1]
    if output_scalar_file_ext not in ['.txt', '.npy']:
        ERROR('Invalid extension for the output scalar file')
    if isfile(options.output_scalar_file) and not options.force:
        ERROR('Output scalar file already exists, use -f to overwrite')

    try:
        # call the actual function
        lengths = compute_lengths(
            options.input_tractogram,
            options.verbose,
        )
        # save the lengths to file
        if output_scalar_file_ext == '.txt':
            np.savetxt(options.output_scalar_file, lengths, fmt='%.4f')
        else:
            np.save(options.output_scalar_file, lengths, allow_pickle=False)

    except Exception as e:
        if isfile(options.output_scalar_file):
            remove(options.output_scalar_file)
        ERROR(e.__str__() if e.__str__() else 'A generic error has occurred')

def tractogram_resample():
    # parse the input parameters
    args = [
        [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output'], {'type': str, 'help': 'Output tractogram'}],
        [['--nb_points', '-n'], {'type': int, 'default': 20, 'help': 'Number of points per streamline'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Overwrite output file if it already exists'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(resample.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.tractogram):
        ERROR(f"Input tractogram file not found: {options.tractogram}")
    if isfile(options.output):
        if options.force:
            WARNING(f"Overwriting output file: {options.output}")
        else:
            ERROR(f"Output file already exists: {options.output}")
    if options.nb_points < 2:
        ERROR(f"Number of points per streamline must be >= 2: {options.nb_points}")

    # call actual function
    resample(
        options.tractogram,
        options.output,
        options.nb_points,
        options.verbose,
        options.force,
    )

def tractogram_sample():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['input_image'], {'type': str, 'help': 'Input image'}],
        [['output_file'], {'type': str, 'help': 'File for the output'}],
        [['--mask', '-m'], {'type': str, 'default': None, 'help': 'Optional mask to restrict the sampling voxels'}],
        [['--space'], {'type': str, 'nargs': '?', 'default': None, 'choices': ['voxmm', 'rasmm', 'vox'], 'help': 'Current reference space of streamlines (rasmm, voxmm, vox), default rasmm'}],
        [['--option'], {'type': str, 'nargs': '?', 'default': 'No_opt', 'choices': ['No_opt', 'mean', 'median', 'min', 'max'], 'help': 'Operation to apply on streamlines, default no operation applied'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(sample.__doc__.split('\n')[0], args)

    # call actual function
    sample(
        options.input_tractogram,
        options.input_image,
        options.output_file,
        options.mask,
        options.space,
        options.option,
        options.force,
        options.verbose
    )

def tractogram_sanitize():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['gray_matter'], {'type': str, 'help': 'Gray matter'}],
        [['white_matter'], {'type': str, 'help': 'White matter'}],
        [['--output_tractogram', '-out'], {'type': str, 'help': 'Output tractogram (if None: "_sanitized" appended to the input filename)'}],
        [['--step'], {'type': float, 'default': 0.2, 'help': 'Step size [in mm]'}],
        [['--max_dist'], {'type': float, 'default': 2, 'help': 'Maximum distance [in mm]'}],
        [['--save_connecting_tck', '-conn'], {'action': 'store_true', 'default': False, 'help': 'Save also tractogram with only the actual connecting streamlines (if True: "_only_connecting" appended to the output filename)'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(sanitize.__doc__.split('\n')[0], args)

    # call actual function
    sanitize(
        options.input_tractogram,
        options.gray_matter,
        options.white_matter,
        options.output_tractogram,
        options.step,
        options.max_dist,
        options.save_connecting_tck,
        options.verbose,
        options.force
    )

def tractogram_smooth():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_tractogram'], {'type': str, 'help': 'Output tractogram'}],
        [['--ratio', '-r'], {'type': float, 'default': 0.25, 'help': 'Ratio of points to be kept/used as control points'}],
        [['--step', '-s'], {'type': float, 'default': 1.0, 'help': 'Sampling step for the output streamlines [in mm]'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(spline_smoothing.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.input_tractogram):
        ERROR(f"Input tractogram file not found: {options.input_tractogram}")
    if isfile(options.output_tractogram) and not options.force:
        ERROR(f"Output tractogram file already exists: {options.output_tractogram}")
    # check if the output tractogram file has the correct extension
    output_tractogram_ext = splitext(options.output_tractogram)[1]
    if output_tractogram_ext not in ['.trk', '.tck']:
        ERROR("Invalid extension for the output tractogram file")

    # check if output tractogram file has absolute path and if not, add the
    # current working directory
    if not isabs(options.output_tractogram):
        options.output_tractogram = p_join(getcwd(), options.output_tractogram)

    # check if ratio and step are valid
    if options.ratio < 0.0 or options.ratio > 1.0:
        ERROR("Invalid ratio, must be between 0 and 1")
    if options.step <= 0.0:
        ERROR("Invalid step, must be greater than 0")

    # call actual function
    spline_smoothing(
        options.input_tractogram,
        options.output_tractogram,
        options.ratio,
        options.step,
        options.verbose,
        options.force
    )

def tractogram_split():
    # parse the input parameters
    args = [
        [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['assignments'], {'type': str, 'help': 'Text file with the streamline assignments'}],
        [['output_folder'], {'type': str, 'nargs': '?', 'default': 'bundles', 'help': 'Output folder for the splitted tractograms'}],
        [['regions'], {'type': int, 'nargs': '*', 'default': [], 'help': 'Streamline connecting the provided region(s) will be extracted'}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': 'Text file with the input streamline weights'}],
        [['--max_open', '-m'], {'type': int, 'help': 'Maximum number of files opened at the same time'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}],
        [['--verbose', '-v'], {'type': int, 'default': 2, 'help': 'Verbose level [0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress]'}]
    ]
    options = setup_parser(split.__doc__.split('\n')[0], args)

    # check if path to input and output files are valid
    if not isfile(options.tractogram):
        ERROR(f"Input tractogram file not found: {options.tractogram}")
    if not isfile(options.assignments):
        ERROR(f"Input assignments file not found: {options.assignments}")
    if not isdir(options.output_folder):
        makedirs(options.output_folder)
    if not isfile(options.assignments):
        ERROR(f"Input assignments file not found: {options.assignments}")
    if options.weights_in is not None:
        if not isfile(options.weights_in):
            ERROR(f"Input weights file not found: {options.weights_in}")
    if options.force:
        WARNING("Overwriting existing files")

    if len(options.regions) > 2:
        ERROR("Too many regions provided, only 2 are allowed")
    if len(options.regions) == 0:
        regions = []
    else:
        regions = [int(i) for i in options.regions]

    # call actual function
    split(
        options.tractogram,
        options.assignments,
        options.output_folder,
        regions,
        options.weights_in,
        options.max_open,
        options.verbose,
        options.force
    )

def tractogram_tsf():
    # parse the input parameters
    args = [
        [['input_tractogram'], {'type': str, 'help': 'Input tractogram'}],
        [['output_tsf'], {'type': str, 'help': 'Output tsf filename'}],
        [['--orientation'], {'action': 'store_true', 'default': False, 'help': 'Color based on orientation'}],
        [['--file'], {'type': str, 'help': 'Color based on given file'}],
        [['--force', '-f'], {'action': 'store_true', 'help': 'Force overwriting of the output'}]
    ]
    options = setup_parser('Create a tsf file for each streamline in order to color them.', args)

    # check if path to input and output files are valid
    if not isfile(options.input_tractogram):
        ERROR(f"Input tractogram file not found: {options.input_tractogram}")
    if isfile(options.output_tsf) and not options.force:
        ERROR(
            f"Output tsf file already exists: {options.output_tsf}, "
            "use -f to overwrite")
    if not options.orientation and not options.file:
        ERROR("Please specify a color option")
    if options.file:
        if not isfile(options.file):
            ERROR(f"Input file not found: {options.file}")

    streamline = LazyTractogram(options.input_tractogram, mode='r')
    num_streamlines = streamline.header['count']

    if options.orientation:
        scalar_arr, n_pts_list = create_color_scalar_file(streamline, int(num_streamlines))
    elif options.file:
        values = np.loadtxt(options.file)
        scalar_arr, n_pts_list = color_by_scalar_file(streamline, values, int(num_streamlines))
    else:
        raise ValueError("Please specify a color option")

    # check if output file exists
    if isfile(options.output_tsf) and not options.force:
        raise IOError("Output file already exists. Use -f to overwrite.")

    tsf = Tsf(options.output_tsf, 'w', header=streamline.header)
    tsf.write_scalar(scalar_arr, n_pts_list)
