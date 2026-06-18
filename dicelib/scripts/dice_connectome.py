from dicelib.ui import setup_logger, setup_parser, get_argparse_info_from_docstring
logger = setup_logger('dice_connectome')


def build():
    '''Entry point for the connectivity.build_connectome function.'''
    from dicelib.connectivity import build_connectome
    summary, desc, notes = get_argparse_info_from_docstring( build_connectome.__doc__  )
    args = [
        [['assignments'], {'type': str, 'help': desc['assignments_filename']}],
        [['out_connectome'], {'type': str, 'help': desc['out_connectome_filename']}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': desc['weights_filename']}],
        [['--stat'], {'type': str, 'default': 'sum', 'help': desc['stat']}],
        [['--symmetric', '-s'], {'action': 'store_true', 'help': desc['symmetric']}],
        [['--tractogram', '-tck'], {'type': str, 'default': None, 'help': desc['tractogram_filename']}],
        [['--atlas', '-a'], {'type': str, 'default': None, 'help': desc['atlas_filename']}],
        [['--distance', '-d'], {'type': float, 'default': 2.0, 'help': desc['distance']}],
        [['--n_threads', '-n'], {'type': int, 'default': 3, 'help': desc['n_threads']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        build_connectome(
            assignments_filename=options.assignments,
            out_connectome_filename=options.out_connectome,
            weights_filename=options.weights_in,
            stat=options.stat,
            symmetric=options.symmetric,
            tractogram_filename=options.tractogram,
            atlas_filename=options.atlas,
            distance=options.distance,
            n_threads=options.n_threads,
            verbose=options.verbose,
            force=options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')
