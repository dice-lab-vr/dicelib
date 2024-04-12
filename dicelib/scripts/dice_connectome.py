from dicelib.connectivity import build_connectome
from dicelib.ui import setup_parser

def connectome_build():
    '''
    Entry point for the connectome build function.
    '''
    # parse the input parameters
    args = [
        [['assignments'], {'type': str, 'help': 'Streamline assignments file (if it doesn\'t exist, it will be created using the given tractogram and atlas files)'}],
        [['connectome_out'], {'type': str, 'help': 'Output connectome file'}],
        [['--weights_in', '-w'], {'type': str, 'default': None, 'help': '''\
                                  Input streamline weights file, used to compute the value of the edges. 
                                  If None, the value of the edges will be number of streamline connecting those regions.'''}],
        [['--metric', '-m'], {'type': str, 'default': 'sum', 'help': '''\
                              Operation to compute the value of the edges, options: sum, mean, min, max. 
                              NB: if \'weights_in\' is None, this parameter is ignored because the connectome will contain the number of streamlines.'''}],
        [['--symmetric', '-s'], {'action': 'store_true', 'help': 'Make output connectome symmetric'}],
        [['--tractogram_in', '-tck'], {'type': str, 'default': None, 'help': '''\
                                     Input tractogram file, used to compute the assignments.
                                     Required if \'assignments_in\' does not exist'''}],
        [['--atlas', '-a'], {'type': str, 'default': None, 'help': '''\
                                Path to the atlas file defining the nodes of the connectome.
                                Required to compute streamlines assignments if \'assignments_in\' does not exist'''}],
        [['--atlas_dist', '-d'], {'type': float, 'default': 2.0, 'help': '''\
                                   Distance used to perform a radial search from each streamline endpoint to locate the nearest node and assign the streamline to the corresponding bundle.
                                   Argument is the maximum radius in mm
                                   Used only if \'assignments_in\' does not exist'''}]
    ]
    options = setup_parser(build_connectome.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    build_connectome(
        options.assignments,
        options.connectome_out,
        options.weights_in,
        options.tractogram_in,
        options.atlas,
        options.atlas_dist,
        options.metric,
        options.symmetric,
        options.verbose,
        options.force
    )
