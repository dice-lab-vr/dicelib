# Change Log
All notable changes will be documented in this file.

## [1.2.1] - 2022-05-02

### Added
- image.pyx module
- dice_image_extract.py script

### Fixed
- Crash when reading long headers
- Few minor bugs

## [1.2.0] - 2022-04-13

### Changed
- Restructuring of files/folders

## [1.1.4] - 2021-12-22

### Fixed
- Handling multiple values in header fields

## [1.1.3] - 2021-12-20

### Added
- ColoredArgParser to provide colored usage/help messages in scripts
- dice_tractogram_split.py: now saves unassigned streamlines in a separate file

### Fixed
- lazytck: small bugs
- Code restructuring

### Changed
- Scripts in bin/ are not all lowercase

## [1.1.2] - 2021-12-02

### Added
- dice_tractogram_info.py: print details about a tractogram
- dice_tractogram_lenghts.py: compute streamline lengths
- ui: added more ANSI color codes

### Fixed
- lazytck: bug when updating file size
- lazytck: avoid creating buffer when tractogram is open for writing

## [1.1.0] - 2021-11-24

### Added
- lazytck module for lazy reading/writing streamlines to/from .TCK tractograms
- dice_tractogram_edit.py script

### Fixed
- spline_smoothing(): error when n_points < 3

## [1.0.0] - 2021-11-11

### Added
- Created first scaffold of the library
