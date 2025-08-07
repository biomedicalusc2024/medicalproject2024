# FSL Wrapper

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/fslwrapper.svg)](https://badge.fury.io/py/fslwrapper)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://fslwrapper.readthedocs.io/)

## Features

- **Modular Design**: Each FSL tool is wrapped in its own module with a consistent interface
- **Extensible Architecture**: Easy to add new FSL tools by inheriting from the base class
- **Robust Error Handling**: Comprehensive exception handling with detailed error messages
- **Parameter Validation**: Automatic validation of input parameters and file paths
- **Pipeline Support**: Built-in pipeline functionality for complex processing workflows
- **Environment Detection**: Automatic FSL environment checking and validation
- **Comprehensive Testing**: Full test suite with unit tests for all components
- **Professional Documentation**: Detailed documentation with usage examples

## Supported FSL Tools

- **BET** (Brain Extraction Tool) - Brain tissue extraction from T1-weighted images
- **FLIRT** (Linear Registration Tool) - Linear image registration
- **FAST** (Segmentation Tool) - Automated tissue segmentation
- **FSLMaths** (Mathematical Operations) - Mathematical operations on images

## Installation

### Prerequisites

- Python 3.7 or higher
- FSL (FMRIB Software Library) installed and configured
- `fslpy` package

### Install FSL Wrapper

```bash
# Install from PyPI
pip install fslwrapper

# Or install from source
git clone https://github.com/your-repo/fslwrapper.git
cd fslwrapper
pip install -e .
```

### Verify Installation

```python
from fslwrapper import FSLWrapper

# Create wrapper instance
wrapper = FSLWrapper()

# Check FSL environment
fsl_info = wrapper.check_environment()
print(f"FSL Version: {fsl_info['version']}")
```

## Quick Start

### Basic Usage

```python
from fslwrapper import FSLWrapper

# Initialize wrapper
wrapper = FSLWrapper()

# Run brain extraction
result = wrapper.run_brain_extraction(
    input_file="subject_t1.nii.gz",
    output_file="subject_t1_brain.nii.gz",
    fractional_intensity=0.5,
    robust=True
)

if result['success']:
    print(f"Brain extraction completed: {result['output_file']}")
else:
    print(f"Brain extraction failed: {result['stderr']}")
```

### Using Individual Tools

```python
from fslwrapper import BET, FLIRT, FAST, FSLMaths

# Brain extraction
bet = BET()
result = bet.extract_brain("input.nii.gz", "output_brain.nii.gz")

# Image registration
flirt = FLIRT()
result = flirt.register_affine("input.nii.gz", "reference.nii.gz", "registered.nii.gz")

# Tissue segmentation
fast = FAST()
result = fast.segment_t1("input.nii.gz", "seg_output", number_classes=3)

# Mathematical operations
math = FSLMaths()
result = math.threshold("input.nii.gz", "thresholded.nii.gz", lower_threshold=0.5)
```

### Pipeline Processing

```python
from fslwrapper import FSLWrapper

wrapper = FSLWrapper()

# Create processing pipeline
pipeline = wrapper.create_pipeline("Complete Brain Processing")

# Add processing steps
pipeline.add_step("Brain Extraction", wrapper.bet, {
    'input_file': 't1.nii.gz',
    'output_file': 't1_brain.nii.gz',
    'fractional_intensity': 0.5,
    'robust': True
})

pipeline.add_step("Registration", wrapper.flirt, {
    'input_file': 't1_brain.nii.gz',
    'reference_file': 'mni152_t1_2mm.nii.gz',
    'output_file': 't1_brain_registered.nii.gz',
    'cost_function': 'mutualinfo',
    'dof': 12
})

pipeline.add_step("Segmentation", wrapper.fast, {
    'input_file': 't1_brain_registered.nii.gz',
    'output_basename': 'seg',
    'number_classes': 3,
    'bias_correction': True
})

# Execute pipeline
results = pipeline.execute(stop_on_error=True)

# Check results
failed_steps = pipeline.get_failed_steps()
if failed_steps:
    print(f"Pipeline failed with {len(failed_steps)} failed steps")
else:
    print("Pipeline completed successfully!")
```

## Project Structure

```
fslwrapper/
│
├── fslwrapper/
│   ├── __init__.py          # Main package entry point
│   ├── core.py              # Abstract base classes
│   ├── bet.py               # BET (Brain Extraction Tool)
│   ├── flirt.py             # FLIRT (Linear Registration Tool)
│   ├── fast.py              # FAST (Segmentation Tool)
│   ├── math.py              # FSLMaths (Mathematical Operations)
│   ├── utils.py             # Utility functions
│   └── exceptions.py        # Custom exception classes
│
├── tests/
│   ├── __init__.py
│   ├── test_core.py         # Core functionality tests
│   ├── test_bet.py          # BET tool tests
│   └── test_flirt.py        # FLIRT tool tests
│
├── examples/
│   └── demo_pipeline.py     # Complete processing pipeline example
│
├── docs/
│   └── usage.md             # Detailed usage documentation
│
├── setup.py                 # Package installation script
├── requirements.txt         # Dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Core Components

### FSLWrapper Class

The main entry point that provides a unified interface to all FSL tools:

```python
from fslwrapper import FSLWrapper

wrapper = FSLWrapper()

# Access individual tools
bet = wrapper.bet
flirt = wrapper.flirt
fast = wrapper.fast
math = wrapper.math

# Convenience methods
result = wrapper.run_brain_extraction("input.nii.gz")
result = wrapper.run_registration("input.nii.gz", "reference.nii.gz")
result = wrapper.run_segmentation("input.nii.gz")
```

### BaseFSLCommand Class

Abstract base class that all FSL tools inherit from, providing:

- Common functionality for all tools
- Parameter validation
- Error handling
- File path validation
- Command execution

### FSLPipeline Class

For complex processing workflows:

```python
pipeline = wrapper.create_pipeline("My Pipeline")
pipeline.add_step("Step 1", tool, parameters)
pipeline.add_step("Step 2", tool, parameters)
results = pipeline.execute()
```

## Error Handling

The package provides comprehensive error handling with custom exceptions:

```python
from fslwrapper.exceptions import (
    FSLWrapperError,
    FSLNotFoundError,
    FSLCommandError,
    FSLParameterError,
    FSLFileError
)

try:
    result = wrapper.run_brain_extraction("input.nii.gz")
except FSLNotFoundError as e:
    print(f"FSL not found: {e}")
except FSLParameterError as e:
    print(f"Invalid parameter: {e}")
except FSLCommandError as e:
    print(f"Command failed: {e}")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e .[test]

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=fslwrapper --cov-report=html
```

## Documentation

- **Usage Guide**: See `docs/usage.md` for detailed usage instructions
- **API Documentation**: Generated from docstrings
- **Examples**: See `examples/demo_pipeline.py` for complete examples

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
cd fslwrapper

pip install -e .[dev]

pytest tests/

flake8 fslwrapper/
black fslwrapper/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex

```

## Acknowledgments

- FSL (FMRIB Software Library) team for the underlying tools
- fslpy developers for the Python interface
- The neuroimaging community for feedback and contributions

## Support

- **Issues**:
- **Documentation**:
- **Email**:

## Changelog

### Version 1.0.0

- Initial release
- Support for BET, FLIRT, FAST, and FSLMaths
- Pipeline processing capabilities
- Comprehensive error handling
- Full test suite
- Complete documentation
