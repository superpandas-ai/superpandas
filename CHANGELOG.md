# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-08-24
### Added
- SQL accessor for pandas DataFrames with SQLite backend
- Support for complex SQL queries including JOINs, aggregations, and string functions
- Multiple table support via environment parameter
- Custom database URI support for persistent storage
- Comprehensive error handling and validation for SQL queries
- Basic langgraph codeagent
### Changed
- Renamed tutorials directory to examples for better organization

## [0.3.4] - 2025-08-03
### Added
- Improved test coverage for serialization and metadata
- Enhanced documentation on readthedocs
### Changed
- Updated README for clarity and usage examples
### Fixed
- Refactored `create_super_dataframe` with parameters
- CSV serialization now preserves index and metadata correctly
### Removed
- LLMClient is no longer user-adjustable

## [0.3.2] - 2025-06-07
### Added
- readthedocs documentation
### Fixed
- column_descriptions API
- logo url


## [0.3.0] - 2025-06-01
### Added
- Beta Release with major refactoring

## [0.2.1] - 2025-04-29

### Added
- Added default column names to 'column_description' and 'column_types' attributes
- Added changelog system

## [0.2.0] - 2025-04-08

### Added
- Initial release of superpandas
- Basic pandas functionality with AI capabilities

## [0.1.0]

### Added
- Initial project setup with basic pandas enhancements 