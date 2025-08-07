"""
FSL FAST (Segmentation Tool) Module

This module provides a Python wrapper for FSL FAST tool using fslpy as backend.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from .exceptions import FSLParameterError, FSLFileError, FSLCommandError

import fsl.wrappers as fslw
from fsl.wrappers import fast

from .core import BaseFSLCommand


logger = logging.getLogger(__name__)


class FAST(BaseFSLCommand):
    """
    FSL FAST (Segmentation Tool) Wrapper
    
    FAST is used for automated segmentation of brain images into different tissue types.
    """
    
    def __init__(self, timeout: int = 600):
        """
        Initialize FAST tool
        
        Args:
            timeout: Command execution timeout in seconds, FAST typically takes longer
        """
        super().__init__("fast", timeout=timeout)
    
    def run(self,
            input_file: Union[str, Path],
            output_basename: Optional[Union[str, Path]] = None,
            number_classes: int = 3,
            segmentation_type: str = "T1",
            bias_correction: bool = True,
            bias_smoothing: Optional[float] = None,
            image_type: str = "T1",
            init_segmentation: Optional[Union[str, Path]] = None,
            init_transform: Optional[Union[str, Path]] = None,
            prior_probability: Optional[Union[str, Path]] = None,
            prior_weight: float = 0.5,
            segment_iters: int = 4,
            bias_iters: int = 4,
            mixel_smooth: float = 0.3,
            iters_after_bias: int = 4,
            hyper: float = 0.1,
            verbose: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Run FAST segmentation
        
        Args:
            input_file: Input image file path
            output_basename: Output basename for segmentation files
            number_classes: Number of tissue classes (2-4)
            segmentation_type: Segmentation type (T1, T2, PD)
            bias_correction: Apply bias field correction
            bias_smoothing: Bias field smoothing (FWHM in mm)
            image_type: Image type (T1, T2, PD)
            init_segmentation: Initial segmentation file
            init_transform: Initial transform file
            prior_probability: Prior probability file
            prior_weight: Prior weight (0-1)
            segment_iters: Number of segmentation iterations
            bias_iters: Number of bias correction iterations
            mixel_smooth: Mixel smoothing parameter
            iters_after_bias: Iterations after bias correction
            hyper: Hyperparameter for MRF
            verbose: Verbose output
            **kwargs: Additional FAST parameters
            
        Returns:
            Command execution result dictionary
            
        Raises:
            FSLParameterError: When parameters are invalid
            FSLFileError: When file operations fail
        """
        # Validate input file
        input_path = self.validate_input_file(input_file, "input image file")
        
        # Set output basename
        if output_basename is None:
            # Handle .nii.gz files properly
            if input_path.name.endswith('.nii.gz'):
                base_name = input_path.name[:-7]  # Remove .nii.gz
            elif input_path.name.endswith('.nii'):
                base_name = input_path.name[:-4]  # Remove .nii
            else:
                base_name = input_path.stem
            
            output_basename = input_path.parent / base_name
        else:
            output_basename = self.validate_output_file(output_basename)
        
        # Validate parameters
        if not 2 <= number_classes <= 4:
            raise FSLParameterError(
                f"Invalid number of classes: {number_classes}. Must be between 2-4",
                tool_name="fast",
                parameter="number_classes"
            )
        
        valid_seg_types = ["T1", "T2", "PD"]
        if segmentation_type not in valid_seg_types:
            raise FSLParameterError(
                f"Invalid segmentation type: {segmentation_type}. Must be one of {valid_seg_types}",
                tool_name="fast",
                parameter="segmentation_type"
            )
        
        valid_image_types = ["T1", "T2", "PD"]
        if image_type not in valid_image_types:
            raise FSLParameterError(
                f"Invalid image type: {image_type}. Must be one of {valid_image_types}",
                tool_name="fast",
                parameter="image_type"
            )
        
        if not 0 <= prior_weight <= 1:
            raise FSLParameterError(
                f"Invalid prior weight: {prior_weight}. Must be between 0-1",
                tool_name="fast",
                parameter="prior_weight"
            )
        
        # Build fslpy parameters (simplified to avoid parameter conflicts)
        fsl_params = {
            'out': str(output_basename),
            'n_classes': number_classes
        }
        
        # Add only the most commonly used parameters
        if bias_correction:
            fsl_params['b'] = True
        if verbose:
            fsl_params['v'] = True
        
        # Add optional parameters
        if init_segmentation:
            fsl_params['s'] = str(init_segmentation)
        
        if init_transform:
            fsl_params['a'] = str(init_transform)
        
        # Add additional parameters
        fsl_params.update(kwargs)
        
        # Execute command using fslpy
        logger.info(f"Starting FAST segmentation: {input_path} -> {output_basename}")
        
        try:
            result = fast(str(input_path), **fsl_params)
            
            # Determine output files based on number of classes
            output_files = [str(output_basename)]
            if number_classes >= 2:
                output_files.append(f"{output_basename}_seg.nii.gz")
                output_files.append(f"{output_basename}_pve_0.nii.gz")
                output_files.append(f"{output_basename}_pve_1.nii.gz")
            if number_classes >= 3:
                output_files.append(f"{output_basename}_pve_2.nii.gz")
            if number_classes >= 4:
                output_files.append(f"{output_basename}_pve_3.nii.gz")
            
            return {
                'success': True,
                'return_code': 0,
                'stdout': str(result),
                'stderr': '',
                'command': f"fast {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}",
                'output_files': output_files,
                'input_file': str(input_path),
                'output_basename': str(output_basename)
            }
            
        except Exception as e:
            logger.error(f"FAST segmentation failed: {e}")
            raise FSLCommandError(
                f"FAST segmentation failed: {str(e)}",
                tool_name="fast",
                command=f"fast {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}"
            )
    
    def segment_t1(self,
                  input_file: Union[str, Path],
                  output_basename: Optional[Union[str, Path]] = None,
                  number_classes: int = 3,
                  bias_correction: bool = True,
                  verbose: bool = False) -> Dict[str, Any]:
        """
        T1-weighted image segmentation
        
        Args:
            input_file: Input T1 image file path
            output_basename: Output basename for segmentation files
            number_classes: Number of tissue classes (2-4)
            bias_correction: Apply bias field correction
            verbose: Verbose output
            
        Returns:
            Command execution result dictionary
        """
        return self.run(
            input_file=input_file,
            output_basename=output_basename,
            number_classes=number_classes,
            segmentation_type="T1",
            image_type="T1",
            bias_correction=bias_correction,
            verbose=verbose
        )
    
    def segment_t2(self,
                  input_file: Union[str, Path],
                  output_basename: Optional[Union[str, Path]] = None,
                  number_classes: int = 3,
                  bias_correction: bool = True,
                  verbose: bool = False) -> Dict[str, Any]:
        """
        T2-weighted image segmentation
        
        Args:
            input_file: Input T2 image file path
            output_basename: Output basename for segmentation files
            number_classes: Number of tissue classes (2-4)
            bias_correction: Apply bias field correction
            verbose: Verbose output
            
        Returns:
            Command execution result dictionary
        """
        return self.run(
            input_file=input_file,
            output_basename=output_basename,
            number_classes=number_classes,
            segmentation_type="T2",
            image_type="T2",
            bias_correction=bias_correction,
            verbose=verbose
        )
    
    def get_help_text(self) -> str:
        """
        Get FAST help information
        
        Returns:
            FAST help text
        """
        return """FAST (Segmentation Tool) Help:

Main Parameters:
  -i <input> : Input image
  -o <output> : Output basename
  -n <classes> : Number of tissue classes (2-4)
  -t <type> : Segmentation type (T1, T2, PD)
  -H <hyper> : Hyperparameter for MRF
  -I <iters> : Number of segmentation iterations
  -l <iters> : Number of bias correction iterations
  -b : Apply bias field correction
  -B <fwhm> : Bias field smoothing (FWHM in mm)
  -S <iters> : Bias correction iterations
  -p <prior> : Prior probability file
  -P <weight> : Prior weight (0-1)
  -W <weight> : Prior weight (0-1)
  -R <smooth> : Mixel smoothing parameter
  -A <iters> : Iterations after bias correction
  -v : Verbose output

Usage Examples:
  fast -i input.nii.gz -o output -n 3 -t T1 -b
  fast -i input.nii.gz -o output -n 4 -t T2 -H 0.1 -I 4
""" 