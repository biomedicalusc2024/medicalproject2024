# FSL Wrapper Usage Guide

This document provides a comprehensive guide on how to use our FSL Wrapper package for neuroimaging analysis.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Tool Usage](#tool-usage)
5. [Pipeline Processing](#pipeline-processing)
6. [Error Handling](#error-handling)
7. [Advanced Usage](#advanced-usage)
8. [Examples](#examples)

## Installation

### Prerequisites

- Python 3.7 or higher
- FSL (FMRIB Software Library) installed and configured
- fslpy package

### Install FSL Wrapper

```bash
pip install fslwrapper
```

Or install from source:

```bash
git clone https://github.com/biomedicalusc2024/medicalproject2024.git
cd fslwrapper
pip install -e .
```

### Verify Installation

```python
from fslwrapper import FSLWrapper

# Create wrapper instance
wrapper = FSLWrapper()

# Check environment
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

## Core Concepts

### FSLWrapper Class

The main entry point for the package is the `FSLWrapper` class, which provides:

- Unified interface to all FSL tools
- Environment checking and validation
- Pipeline creation and management
- Convenience methods for common operations

### BaseFSLCommand Class

All FSL tools inherit from `BaseFSLCommand`, which provides:

- Common functionality for all tools
- Parameter validation
- Error handling
- File path validation
- Command execution

### FSLPipeline Class

For complex processing workflows, use the `FSLPipeline` class:

```python
from fslwrapper import FSLWrapper

wrapper = FSLWrapper()
pipeline = wrapper.create_pipeline("My Processing Pipeline")

# Add processing steps
pipeline.add_step("Brain Extraction", wrapper.bet, {
    'input_file': 'input.nii.gz',
    'output_file': 'brain.nii.gz',
    'robust': True
})

pipeline.add_step("Registration", wrapper.flirt, {
    'input_file': 'brain.nii.gz',
    'reference_file': 'template.nii.gz',
    'output_file': 'registered.nii.gz'
})

# Execute pipeline
results = pipeline.execute()
```

## Tool Usage

### BET (Brain Extraction Tool)

```python
from fslwrapper import BET

bet = BET()

# Basic brain extraction
result = bet.extract_brain(
    input_file="t1.nii.gz",
    output_file="t1_brain.nii.gz",
    fractional_intensity=0.5,
    robust=True
)

# Advanced brain extraction with all parameters
result = bet.run(
    input_file="t1.nii.gz",
    output_file="t1_brain.nii.gz",
    fractional_intensity=0.3,
    vertical_gradient=0.1,
    center_of_gravity=(90, 100, 130),
    radius=75,
    robust=True,
    remove_eyes=True,
    reduce_bias=True,
    mask=True
)
```

### FLIRT (Linear Registration Tool)

```python
from fslwrapper import FLIRT

flirt = FLIRT()

# Affine registration
result = flirt.register_affine(
    input_file="subject.nii.gz",
    reference_file="template.nii.gz",
    output_file="registered.nii.gz",
    cost_function="mutualinfo",
    dof=12
)

# Rigid body registration
result = flirt.register_rigid(
    input_file="subject.nii.gz",
    reference_file="template.nii.gz",
    output_file="registered.nii.gz",
    cost_function="corratio"
)

# Advanced registration with search parameters
result = flirt.run(
    input_file="subject.nii.gz",
    reference_file="template.nii.gz",
    output_file="registered.nii.gz",
    cost_function="mutualinfo",
    search_range=(-10, 10),
    search_angles=(-5, 5),
    dof=12,
    interp="sinc"
)
```

### FAST (Segmentation Tool)

```python
from fslwrapper import FAST

fast = FAST()

# T1-weighted segmentation
result = fast.segment_t1(
    input_file="t1.nii.gz",
    output_basename="seg",
    number_classes=3,
    bias_correction=True
)

# T2-weighted segmentation
result = fast.segment_t2(
    input_file="t2.nii.gz",
    output_basename="seg_t2",
    number_classes=4,
    bias_correction=True
)

# Advanced segmentation
result = fast.run(
    input_file="t1.nii.gz",
    output_basename="seg",
    number_classes=3,
    segmentation_type="T1",
    bias_correction=True,
    bias_smoothing=20,
    prior_weight=0.5,
    hyper=0.1
)
```

### FSLMaths (Mathematical Operations)

```python
from fslwrapper import FSLMaths

math = FSLMaths()

# Thresholding
result = math.threshold(
    input_file="input.nii.gz",
    output_file="thresholded.nii.gz",
    lower_threshold=0.5,
    upper_threshold=1.0
)

# Smoothing
result = math.smooth(
    input_file="input.nii.gz",
    output_file="smoothed.nii.gz",
    sigma=2.0,
    kernel_type="gaussian"
)

# Mathematical operations
result = math.add("input1.nii.gz", "input2.nii.gz", "sum.nii.gz")
result = math.subtract("input1.nii.gz", "input2.nii.gz", "diff.nii.gz")
result = math.multiply("input.nii.gz", 2.0, "scaled.nii.gz")
result = math.divide("input.nii.gz", 2.0, "divided.nii.gz")

# Masking
result = math.mask("input.nii.gz", "mask.nii.gz", "masked.nii.gz")

# Binarization
result = math.binarize("input.nii.gz", "binary.nii.gz", threshold=0.5)

# Custom operations
result = math.run(
    input_file="input.nii.gz",
    output_file="processed.nii.gz",
    operations=["-thr", "0.5", "-bin", "-s", "2.0"]
)
```

## Pipeline Processing

### Creating Complex Pipelines

```python
from fslwrapper import FSLWrapper

wrapper = FSLWrapper()

# Create a complete processing pipeline
pipeline = wrapper.create_pipeline("Complete Brain Processing")

# Step 1: Brain extraction
pipeline.add_step("Brain Extraction", wrapper.bet, {
    'input_file': 't1.nii.gz',
    'output_file': 't1_brain.nii.gz',
    'fractional_intensity': 0.5,
    'robust': True
})

# Step 2: Registration to template
pipeline.add_step("Registration", wrapper.flirt, {
    'input_file': 't1_brain.nii.gz',
    'reference_file': 'mni152_t1_2mm.nii.gz',
    'output_file': 't1_brain_registered.nii.gz',
    'cost_function': 'mutualinfo',
    'dof': 12
})

# Step 3: Tissue segmentation
pipeline.add_step("Segmentation", wrapper.fast, {
    'input_file': 't1_brain_registered.nii.gz',
    'output_basename': 'seg',
    'number_classes': 3,
    'bias_correction': True
})

# Step 4: Create brain mask
pipeline.add_step("Create Mask", wrapper.math, {
    'input_file': 'seg_seg.nii.gz',
    'output_file': 'brain_mask.nii.gz',
    'operations': ['-thr', '1', '-bin']
})

# Execute pipeline
results = pipeline.execute(stop_on_error=True)

# Check results
failed_steps = pipeline.get_failed_steps()
if failed_steps:
    print(f"Pipeline failed with {len(failed_steps)} failed steps")
    for step in failed_steps:
        print(f"  - {step['step_name']}: {step['result'].get_error_message()}")
else:
    print("Pipeline completed successfully!")
```

### Pipeline with Error Handling

```python
from fslwrapper import FSLWrapper, FSLWrapperError

wrapper = FSLWrapper()

try:
    pipeline = wrapper.create_pipeline("Robust Processing")

    # Add steps with error handling
    pipeline.add_step("Step 1", wrapper.bet, {'input_file': 'input.nii.gz'})
    pipeline.add_step("Step 2", wrapper.flirt, {'input_file': 'brain.nii.gz'})

    # Execute with error handling
    results = pipeline.execute(stop_on_error=False)

    # Check for failed steps
    failed_steps = pipeline.get_failed_steps()
    if failed_steps:
        print("Some steps failed, but pipeline continued:")
        for step in failed_steps:
            print(f"  - {step['step_name']}: {step['result'].get_error_message()}")

except FSLWrapperError as e:
    print(f"Pipeline error: {e}")
```

## Error Handling

### Exception Types

The package defines several custom exceptions:

- `FSLWrapperError`: Base exception for all FSL wrapper errors
- `FSLNotFoundError`: FSL environment not found or not configured
- `FSLCommandError`: FSL command execution failed
- `FSLParameterError`: Invalid parameters provided
- `FSLFileError`: File operation failed

### Error Handling Examples

```python
from fslwrapper import FSLWrapper
from fslwrapper.exceptions import FSLNotFoundError, FSLCommandError, FSLParameterError

wrapper = FSLWrapper()

try:
    # Check environment first
    fsl_info = wrapper.check_environment()
    print(f"FSL environment OK: {fsl_info['version']}")

    # Run processing
    result = wrapper.run_brain_extraction("input.nii.gz")

except FSLNotFoundError as e:
    print(f"FSL not found: {e}")
    print("Please install and configure FSL")

except FSLParameterError as e:
    print(f"Invalid parameter: {e}")
    print(f"Parameter: {e.parameter}")

except FSLCommandError as e:
    print(f"Command failed: {e}")
    print(f"Return code: {e.return_code}")
    print(f"Command: {e.command}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### Validation and Error Prevention

```python
from fslwrapper import FSLWrapper
from pathlib import Path

wrapper = FSLWrapper()

# Validate input files before processing
input_file = Path("input.nii.gz")
if not input_file.exists():
    print(f"Input file not found: {input_file}")
    exit(1)

# Check file format
if not wrapper.utils.is_nifti_file(input_file):
    print(f"Input file is not a valid NIfTI file: {input_file}")
    exit(1)

# Run processing with validation
try:
    result = wrapper.run_brain_extraction(str(input_file))
    print("Processing completed successfully")
except Exception as e:
    print(f"Processing failed: {e}")
```

## Advanced Usage

### Custom Tool Extensions

You can extend the package by creating custom tools:

```python
from fslwrapper.core import BaseFSLCommand

class CustomTool(BaseFSLCommand):
    def __init__(self, timeout=300):
        super().__init__("custom_tool", timeout=timeout)

    def run(self, input_file, output_file, **kwargs):
        # Implement custom tool logic
        # Validate parameters
        # Execute command
        # Return result
        pass

# Use custom tool
custom = CustomTool()
result = custom.run("input.nii.gz", "output.nii.gz")
```

### Batch Processing

```python
from fslwrapper import FSLWrapper
from pathlib import Path
import concurrent.futures

wrapper = FSLWrapper()

def process_subject(subject_dir):
    """Process a single subject"""
    subject_dir = Path(subject_dir)
    t1_file = subject_dir / "t1.nii.gz"

    if not t1_file.exists():
        return f"Subject {subject_dir.name}: T1 file not found"

    try:
        result = wrapper.run_brain_extraction(str(t1_file))
        return f"Subject {subject_dir.name}: Success"
    except Exception as e:
        return f"Subject {subject_dir.name}: Failed - {e}"

# Process multiple subjects in parallel
subject_dirs = [d for d in Path("subjects").iterdir() if d.is_dir()]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_subject, subject_dirs))

for result in results:
    print(result)
```

### Configuration Management

```python
from fslwrapper import FSLWrapper
import json

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Create wrapper with configuration
wrapper = FSLWrapper(timeout=config.get("timeout", 300))

# Use configuration in processing
for subject in config["subjects"]:
    result = wrapper.run_brain_extraction(
        input_file=subject["t1_file"],
        output_file=subject["brain_file"],
        fractional_intensity=config["bet"]["fractional_intensity"],
        robust=config["bet"]["robust"]
    )
```

## Examples

### Complete Processing Workflow

See the `examples/demo_pipeline.py` file for a complete example of a neuroimaging processing pipeline.

### Configuration File Example

```json
{
  "timeout": 600,
  "bet": {
    "fractional_intensity": 0.5,
    "robust": true,
    "remove_eyes": true
  },
  "flirt": {
    "cost_function": "mutualinfo",
    "dof": 12,
    "interp": "sinc"
  },
  "fast": {
    "number_classes": 3,
    "bias_correction": true,
    "hyper": 0.1
  },
  "subjects": [
    {
      "id": "sub-01",
      "t1_file": "data/sub-01/t1.nii.gz",
      "brain_file": "output/sub-01/t1_brain.nii.gz"
    }
  ]
}
```

### Logging Configuration

```python
import logging
from fslwrapper import FSLWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fsl_processing.log'),
        logging.StreamHandler()
    ]
)

# Create wrapper with logging
wrapper = FSLWrapper()

# Processing will now be logged
result = wrapper.run_brain_extraction("input.nii.gz", "output.nii.gz")
```

This usage guide covers the main features and capabilities of the FSL Wrapper package. For more detailed information about specific tools or advanced usage patterns, refer to the individual tool documentation or the source code.
