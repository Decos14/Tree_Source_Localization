# Changelog

All notable changes to this project will be documented in this file.

## [0.6.1] - 2025-07-29
- fixed sample of positive normal to return positive normal instead of normal

## [0.6.0] - 2025-07-29
- built a joint mgf augmentation registry
- cleaned up code in cond_joint_mgf using it

## [0.5.3] - 2025-07-29

### Changed
- cleaned up old comments
- added type hint to localize
- used better logic with None

## [0.5.2] - 2025-07-29

### Changed
- cleaned up variable names

## [0.5.1] - 2025-07-29

### Changed
- Cleaned up build_A_matrix logic

## [0.5.0] - 2025-07-29

### Changed
- Renamed Infection_Simulation to simulate_infection
- Renamed Equivalent_Class to get_equivalent_class

## [0.4.0] - 2025-07-29

### Changed
- Slightly modified JSON input file

## [0.3.1] - 2025-07-29

### Changed
- Rewrote build_connection_tree to be cleaner

## [0.3.0] - 2025-07-29

### Changed
- Input file format is now a JSON instead of a CSV

## [0.2.1] - 2025-07-27

### Changed
- Refactored tree class for fewer instance variables

## [0.2.0] - 2025-07-27

### Changed
- Refactored the edge distributions into it's own abstract and child classes with a distribution registry
- Major refactor of tests to handle the new change

## [0.1.1] - 2025-7-24

### Changed
- Refactors the search algorithm into it's own class
- Added lazy caching to the search algorithm

## [0.1.0] - 2025-07-20

### Added
- Initial release of `tree_source_localization` package.
- Core functionality implemented in `Tree.py` and `MGF_Functions.py`.
- Support for source localization algorithms on tree structures.
- Unit tests and regression tests included under `tests/`.
- `pyproject.toml` configured with dependencies: numpy, scipy.
- Package structured using `src/` layout for clean modularization.
- GPL-3.0-or-later license.