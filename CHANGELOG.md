# Change Log
All notable changes will be documented in this file.

## [1.1.2] - 2021-02-12

### Added
- DICE_tractogram_info.py: print details about a tractogram
- DICE_tractogram_lenghts.py: compute streamline lengths
- ui: added more ANSI color codes

### Fixed
- lazytck: bug when updating file size
- lazytck: avoid creating buffer when tractogram is open for writing

## [1.1.0] - 2021-24-11

### Added
- lazytck module for lazy reading/writing streamlines to/from .TCK tractograms
- DICE_tractogram_edit.py script

### Fixed
- spline_smoothing(): error when n_points < 3

## [1.0.0] - 2021-11-11

### Added
- Created first scaffold of the library
