import dicelib.tsf
from dicelib.ui import setup_logger, setup_parser, get_argparse_info_from_docstring
logger = setup_logger('dice_tsf')


def create():
    '''Entry point for the tsf.create function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tsf.create.__doc__ )
    args = [
        [['tractogram'], {'type': str, 'help': desc['tractogram_filename']}],
        [['scalars'], {'type': str, 'help': desc['scalars_filename']}],
        [['out_tsf'], {'type': str, 'help': desc['out_tsf_filename']}],
        [['--check_orientation', '-check'], {'action': 'store_true', 'default': False, 'help': desc['check_orientation']}],
        [['--out_tractogram'], {'type': str, 'default': None, 'help': desc['out_tractogram_filename']}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tsf.create(
            tractogram_filename=options.tractogram,
            scalars_filename=options.scalars,
            out_tsf_filename=options.out_tsf,
            check_orientation=options.check_orientation,
            out_tractogram_filename=options.out_tractogram,
            force=options.force,
            verbose=options.verbose
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')


def join():
    '''Entry point for the tsf.join function'''
    summary, desc, notes = get_argparse_info_from_docstring( dicelib.tsf.join.__doc__ )
    args = [
        [['tsf_in'], {'type': str, 'nargs': '+', 'help': 'Input tsf files'}],
        [['tsf_out'], {'type': str, 'help': 'Output tsf file'}]
    ]
    options = setup_parser(summary, args, epilog=notes, add_force=True, add_verbose=True)
    try:
        dicelib.tsf.join(
            options.tsf_in,
            options.tsf_out,
            options.verbose,
            options.force
        )
    except Exception as e:
        logger.error(e.__str__() if e.__str__() else 'A generic error has occurred')
