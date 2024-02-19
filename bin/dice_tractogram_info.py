from os.path import isfile
from dicelib.tractogram import info
from dicelib.ui import ColoredArgParser, ERROR

# parse the input parameters
parser = ColoredArgParser(description=info.__doc__.split('\n')[0])
args = [
    [['tractogram'], {'type': str, 'help': 'Input tractogram'}],
    [['--lenghts', '-l'], {'action': 'store_true', 'help': 'Show stats on streamline lenghts'}],
    [['--max_field_length', '-m'], {'type': int, 'help': 'Maximum length allowed for printing a field value'}]
]
for arg in args:
    parser.add_argument(*arg[0], **arg[1])

options = parser.parse_args()

# check if path to input and output files are valid
if not isfile(options.tractogram):
    ERROR(f'Input tractogram file not found: {options.tractogram}')

# call actual function
info(
    options.tractogram,
    options.lenghts,
    options.max_field_length
)
