# Change Log
### All notable changes to `DICElib` will be documented in this file.

## `v1.3.0`<br>_2026-01-14_
### 🐛Fixed
- Conversion from streamline space to voxel space
- Minor bugs

### ✨Added
- Computation of the coherence of streamline with the underlying FOD  (`dice_tractogram_coherence` script)
- Computation of TDI maps  (`dice_tractogram_tdi` script)
- Streamline smoothing with Savitzky–Golay filter
- Support for handling additional TSF files when joining and filtering tractograms

### 🛠️Changed
- Documentation of entry-point scripts is automatically taken from docstrings of corresponding functions
- Function parameters now follow the same conventions
- Minor modifications towards a consistent way to compute voxel coordinates
- Renamed spline smoothing functions
- Removed unused functions

---
---

## `v1.2.1`<br>_2025-10-06_
### 🐛Fixed
- Fixed ending points in `smooth`
- Fix minor bugs (when using jupyter notebooks, windows, and/or newer python versions)

---
---

## `v1.2.0`<br>_2025-01-20_
### ✨Added
- Add `closest_centroid_pt` and `project_values_on_centroid` functions
- Add `check_orientation` option to `tsf_create` function
- Add `cp313` wheels

### 🛠️Changed
- Support for `numpy` version `>=2.0.0`

### 🐛Fixed
- Check on number of points for resample in `smooth`
- Copy streamline values in `tsf_create` function
- Add conditional logic for output tractogram in `tsf_create` function
- Fix `clust_idx` returned by `run_clustering` when using atlas and add option to save them
- Fix `SyntaxWarning` in `dicelib.ui.ascii_art`

---
---

## `v1.1.2`<br>_2024-07-04_
### ✨Added
- Added new options for smoothing
- Added function to save the replicas used for the blur
- Added function to shuffle the tractogram (`dice_tractogram_shuffle` script)
- Add function to compute the TDI of the ending points, possibly using blur (`dice_image_tdi_ends` script)
- Precompiled wheels for Windows, MacOS, and Linux are now available on PyPI

### 🛠️Changed
- Restict the `numpy` version to `<2.0.0`

### 🐛Fixed
- Errors in `dice_tractogram_filter` script
- Weights grouping when perform clustering
- Improved output when running from Jupyter notebook

---
---

## `v1.1.1`<br>_2024-04-12_
### 🐛Fixed
- Fixed typo in `dice_connectome.py` script

---
---

## `v1.1.0`<br>_2024-04-12_
### 🛠️Changed
- Requires Python>=3.8
- Store project metadata in `pyproject.toml` (PEP 621)
- Reformat scripts as `entry points` (executable commands)
- Refactor `ui.py` module
- Improved functions

### ✨Added
- Added some functions

---
---

## `v1.0.4`<br>_2023-11-08_
### 🐛Fixed
- Fixed connectivity radial search assignments

---
---

## `v1.0.3`<br>_2023-10-31_

### ✨Added
- `_in_notebook()` method to check if code is running in a Jupyter notebook

---
---

## `v1.0.2`<br>_2023-10-16_

### 🐛Fixed
- Fixed help descriptions
- Removed unused `test_smooth.py` script

---
---

## `v1.0.1`<br>_2023-09-21_

### 🐛Fixed
- Removed unused control in `dice_tractogram_filter.py`
- Removed unused `splines` module from `streamline.pyx`

---
---

## `v1.0.0`<br>_2023-09-14_
🎉published on PyPI🎉
### 🛠️Changed
- Switched to proprietary license (see `LICENSE` file)

### ✨Added
- `ui.ProgressBar()`
    - Indeterminate progress bar
    - Determinate progress bar with support for multithreading

### 🐛Fixed

---
---

## [1.3.0] - 2023-02-16

### Changed
- lazytck renamed lazytractogram

## [1.2.1] - 2022-05-02

### Added
- image.pyx module
- dice_image_extract.py script
- dice_tractogram_filter.py: possibility to remove streamlines randomly

### Fixed
- Crash when reading long headers
- Few minor bugs

### Changed
- Added 4th verbosity value

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
