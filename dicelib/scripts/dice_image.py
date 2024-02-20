from dicelib.image import extract
from dicelib.ui import ColoredArgParser, ERROR

from os.path import isfile

def image_extract():
    # parse the input parameters
    parser = ColoredArgParser(description=extract.__doc__.split('\n')[0])
    args = [
        [["input_dwi"], {"help": "Input DWI data"}],
        [["input_scheme"], {"help": "Input scheme"}],
        [["output_dwi"], {"help": "Output DWI data"}],
        [["output_scheme"], {"help": "Output scheme"}],
        [["--b", "-b"], {"type": float, "nargs": '+', "required": True, "help": "List of b-values to extract"}],
        [["--round", "-r"], {"type": float, "default": 0.0, "help": "Round b-values to nearest integer multiple of this value"}],
        [["--verbose", "-v"], {"default": 2, "type": int, "help": "Verbose level [ 0 = no output, 1 = only errors/warnings, 2 = errors/warnings and progress, 3 = all messages, no progress, 4 = all messages and progress ]"}],
        [["--force", "-f"], {"action": "store_true", "help": "Force overwriting of the output"}]
    ]
    for arg in args:
        parser.add_argument(*arg[0], **arg[1])
    options = parser.parse_args()

    # check if path to input and output files are valid
    if not isfile(options.input_dwi):
        ERROR(f"Input DWI data file not found: {options.input_dwi}")
    if not isfile(options.input_scheme):
        ERROR(f"Input scheme file not found: {options.input_scheme}")
    if isfile(options.output_dwi) and not options.force:
        ERROR(f"Output DWI data file already exists: {options.output_dwi} "
                "use -f to overwrite")
    if isfile(options.output_scheme) and not options.force:
        ERROR(f"Output scheme file already exists: {options.output_scheme} "
                "use -f to overwrite")

    # check if b-values and round are valid
    if options.round < 0.0:
        ERROR(f"Round value must be >= 0.0: {options.round}")
    if len(options.b) == 0:
        ERROR("No b-values specified")
    for b in options.b:
        if b < 0.0:
            ERROR(f"b-value must be >= 0.0: {b}")

    # call actual function
    extract(
        options.input_dwi,
        options.input_scheme,
        options.output_dwi,
        options.output_scheme,
        options.b,
        options.round,
        options.verbose,
        options.force
    )
