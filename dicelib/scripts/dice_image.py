from dicelib.image import extract, tdi_ends
from dicelib.ui import setup_parser

def image_extract():
    '''
    Entry point for the image extract function.
    '''
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

def image_tdi_ends():
    '''
    Entry point for the tdi of ending points (with blur) function.
    '''
    # parse the input parameters
    args = [
        [["input_tractogram"], {"help": "Input tractogram"}],
        [["input_ref"], {"help": "Input reference image"}],
        [["output_image"], {"help": "Output image"}],
        [["--blur_core_extent", "-core"], {"type": float, "default": 0.0, "help": "Core extent for blurring (in mm)"}],
        [["--blur_gauss_extent", "-gauss"], {"type": float, "default": 0.0, "help": "Gaussian extent for blurring (in mm)"}],
        [["--blur_spacing", "-spacing"], {"type": float, "default": 0.25, "help": "Spacing for blurring (in mm)"}],
        [["--blur_gauss_min", "-min"], {"type": float, "default": 0.1, "help": "Minimum Gaussian value for blurring (in mm)"}],
        [["--fiber_shift", "-shift"], {"type": float, "default": 0.0, "help": '''\
                                       If necessary, shift the streamline coordinates by this amount. 
                                       Either a single value or a list of three values.
                                       The value is specified in voxel units, e.g., 0.5 translates by half voxel.'''}]
    ]
    options = setup_parser(extract.__doc__.split('\n')[0], args, add_force=True, add_verbose=True)

    # call actual function
    tdi_ends(
        options.input_tractogram, 
        options.input_ref, 
        options.output_image, 
        options.blur_core_extent,
        options.blur_gauss_extent,
        options.blur_spacing,
        options.blur_gauss_min,
        options.fiber_shift, 
        options.verbose,
        options.force
    )
