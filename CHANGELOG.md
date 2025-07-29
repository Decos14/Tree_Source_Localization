# Changelog

All notable changes to this project will be documented in this file.

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