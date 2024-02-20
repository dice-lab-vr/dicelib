from dicelib.connectome_blur import build_connectome
from dicelib.ui import ColoredArgParser, ERROR

from os import getcwd
from os.path import isabs, isfile, join as p_join, splitext

def connectome_build():
    # parse the input parameters
    parser = ColoredArgParser(description=build_connectome.__doc__.split('\n')[0])
    args = [
        [["input_assignments"], {"help": "Input streamline assignments file"}],
        [["output_connectome"], {"help": "Output connectome"}],
        [["input_weights"], {"help": "Input streamline weights file"}],
        [["--metric", "-m"], {"default": 'sum', "help": "Operation to compute the value of the edges, options: sum, mean, min, max."}],
        [["--symmetric", "-s"], {"action": "store_true", "help": "Make output connectome symmetric"}],
        [["--verbose", "-v"], {"default": 2, "type": int, "help": "Verbose level [ 0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress ]"}],
        [["--force", "-f"], {"action": "store_true", "help": "Force overwriting of the output"}]
    ]
    for arg in args:
        parser.add_argument(*arg[0], **arg[1])
    options = parser.parse_args()


    # check if path to input and output files are valid
    if not isfile(options.input_assignments):
        ERROR(f"Input assignments file not found: {options.input_assignments}")
    if isfile(options.output_connectome) and not options.force:
        ERROR(f"Output conncetome file already exists: {options.output_connectome}")
    # check if the output connectome file has the correct extension
    output_connectome_ext = splitext(options.output_connectome)[1]
    if output_connectome_ext not in ['.csv', '.npy']:
        ERROR("Invalid extension for the output connectome file")

    # check if the output connectome file has absolute path and if not, add the current working directory
    if options.output_connectome and not isabs(options.output_connectome):
        options.output_connectome = p_join(getcwd(), options.output_connectome)

    # call actual function
    build_connectome( 
        options.input_assignments,
        options.output_connectome,
        options.input_weights, 
        options.metric, 
        options.symmetric,
        options.verbose,
        options.force
    )
