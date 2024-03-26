from dicelib.connectivity import build_connectome
from dicelib.ui import setup_parser

def connectome_build():
    # parse the input parameters
    args = [
        [['assignments_in'], {'type': str, 'help': 'Input streamline assignments file'}],
        [['connectome_out'], {'type': str, 'help': 'Output connectome file'}],
        [['weights_in'], {'type': str, 'help': 'Input streamline weights file, used to compute the value of the edges'}],
        [['--tractogram_in', '-t'], {'type': str, 'help': '''\
                                     Input tractogram file, used to compute the assignments.
                                     Required if \'assignments_in\' does not exist'''}],
        [['--nodes_in', '-n'], {'type': str, 'help': '''\
                                Input nodes file, used to compute the assignments.
                                Required if \'assignments_in\' does not exist'''}],
        [['--threshold', '-thr'], {'type': float, 'default': 2.0, 'help': '''\Threshold used to compute the assignments.
                                   Required if \'assignments_in\' does not exist'''}],
        [['--metric', '-m'], {'type': str, 'default': 'sum', 'help': 'Operation to compute the value of the edges, options: sum, mean, min, max.'}],
        [['--symmetric', '-s'], {'action': 'store_true', 'help': 'Make output connectome symmetric'}]
    ]
    options = setup_parser(build_connectome.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    build_connectome(
        options.weights_in,
        options.assignments_in,
        options.connectome_out,
        options.tractogram_in,
        options.nodes_in,
        options.threshold,
        options.metric,
        options.symmetric,
        options.verbose,
        options.force
    )
