from dicelib.image import extract
from dicelib.ui import setup_parser

def image_extract():
    # parse the input parameters
    args = [
        [["dwi_in"], {"help": "Input DWI data"}],
        [["scheme_in"], {"help": "Input scheme"}],
        [["dwi_out"], {"help": "Output DWI data"}],
        [["scheme_out"], {"help": "Output scheme"}],
        [["--b", "-b"], {"type": float, "nargs": '+', "metavar": "B", "required": True, "help": "List of b-values to extract"}],
        [["--round", "-r"], {"type": float, "default": 0.0, "help": "Round b-values to nearest integer multiple of this value"}]
    ]
    options = setup_parser(extract.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    extract(
        options.dwi_in,
        options.scheme_in,
        options.dwi_out,
        options.scheme_out,
        options.b,
        options.round,
        options.verbose,
        options.force
    )
