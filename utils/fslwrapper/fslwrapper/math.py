"""
FSL FSLMaths (Mathematical Operations Tool) Module

This module provides a Python wrapper for FSL FSLMaths tool using fslpy as backend.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from .exceptions import FSLParameterError, FSLFileError, FSLCommandError

import fsl.wrappers as fslw
from fsl.wrappers import fslmaths

from .core import BaseFSLCommand


logger = logging.getLogger(__name__)


class FSLMaths(BaseFSLCommand):
    """
    FSL FSLMaths (Mathematical Operations Tool) Wrapper
    
    FSLMaths is used for mathematical operations on images.
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize FSLMaths tool
        
        Args:
            timeout: Command execution timeout in seconds
        """
        super().__init__("fslmaths", timeout=timeout)
    
    def run(self,
            input_file: Union[str, Path],
            output_file: Optional[Union[str, Path]] = None,
            operations: Optional[List[str]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Run FSLMaths operations
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            operations: List of mathematical operations to apply
            **kwargs: Additional FSLMaths parameters
            
        Returns:
            Command execution result dictionary
            
        Raises:
            FSLParameterError: When parameters are invalid
            FSLFileError: When file operations fail
        """
        # Validate input file
        input_path = self.validate_input_file(input_file, "input image file")
        
        # Set output file path
        if output_file is None:
            input_stem = input_path.stem
            if input_stem.endswith('.nii'):
                input_stem = input_stem[:-4]  # Remove .nii extension from stem
            output_path = input_path.parent / f"{input_stem}_processed{input_path.suffix}"
        else:
            output_path = self.validate_output_file(output_file)
        
        # Validate operations parameter
        if operations is not None and not isinstance(operations, list):
            raise FSLParameterError(
                "operations must be a list of strings",
                tool_name="fslmaths",
                parameter="operations"
            )
        
        # Execute command using fslpy
        logger.info(f"Starting FSLMaths operations: {input_path} -> {output_path}")
        
        try:
            # Create fslmaths object
            math_obj = fslmaths(str(input_path))
            
            # Apply operations if provided
            if operations:
                for i in range(0, len(operations), 2):
                    op = operations[i].lstrip('-')
                    if i + 1 < len(operations):
                        arg = operations[i + 1]
                        if hasattr(math_obj, op):
                            if op in ['thr', 'uthr', 'mul', 'add', 'sub', 'div']:
                                math_obj = getattr(math_obj, op)(float(arg))
                            else:
                                math_obj = getattr(math_obj, op)(arg)
                    else:
                        if hasattr(math_obj, op):
                            math_obj = getattr(math_obj, op)()
            
            # Run the operations and save to output file
            result = math_obj.run(str(output_path))
            
            return {
                'success': True,
                'return_code': 0,
                'stdout': str(result),
                'stderr': '',
                'command': f"fslmaths {str(input_path)} {' '.join(operations) if operations else ''} {str(output_path)}",
                'output_file': str(output_path),
                'input_file': str(input_path)
            }
            
        except Exception as e:
            logger.error(f"FSLMaths operations failed: {e}")
            raise FSLCommandError(
                f"FSLMaths operations failed: {str(e)}",
                tool_name="fslmaths",
                command=f"fslmaths {str(input_path)} {' '.join(operations) if operations else ''} {str(output_path)}"
            )
    
    def threshold(self,
                 input_file: Union[str, Path],
                 output_file: Optional[Union[str, Path]] = None,
                 lower_threshold: float = 0,
                 upper_threshold: Optional[float] = None,
                 use_robust: bool = False) -> Dict[str, Any]:
        """
        Apply thresholding to image
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            lower_threshold: Lower threshold value
            upper_threshold: Upper threshold value (optional)
            use_robust: Use robust thresholding
            
        Returns:
            Command execution result dictionary
        """
        operations = []
        
        if use_robust:
            operations.append(f"-thrP {lower_threshold}")
        else:
            operations.append(f"-thr {lower_threshold}")
        
        if upper_threshold is not None:
            operations.append(f"-uthr {upper_threshold}")
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def smooth(self,
              input_file: Union[str, Path],
              output_file: Optional[Union[str, Path]] = None,
              sigma: float = 1.0,
              kernel_type: str = "gaussian") -> Dict[str, Any]:
        """
        Apply smoothing to image
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            sigma: Smoothing sigma (FWHM)
            kernel_type: Kernel type (gaussian, mean, median)
            
        Returns:
            Command execution result dictionary
        """
        if kernel_type == "gaussian":
            operations = [f"-s {sigma}"]
        elif kernel_type == "mean":
            operations = [f"-kernel mean {sigma}"]
        elif kernel_type == "median":
            operations = [f"-kernel median {sigma}"]
        else:
            raise FSLParameterError(
                f"Invalid kernel type: {kernel_type}. Must be gaussian, mean, or median",
                tool_name="fslmaths",
                parameter="kernel_type"
            )
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def mask(self,
            input_file: Union[str, Path],
            mask_file: Union[str, Path],
            output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Apply mask to image
        
        Args:
            input_file: Input image file path
            mask_file: Mask image file path
            output_file: Output image file path
            
        Returns:
            Command execution result dictionary
        """
        # Validate mask file
        mask_path = self.validate_input_file(mask_file, "mask image file")
        
        operations = [f"-mas {mask_path}"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def add(self,
           input_file: Union[str, Path],
           value_or_file: Union[float, str, Path],
           output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Add value or image to input
        
        Args:
            input_file: Input image file path
            value_or_file: Value to add or image file path
            output_file: Output image file path
            
        Returns:
            Command execution result dictionary
        """
        if isinstance(value_or_file, (int, float)):
            operations = [f"-add {value_or_file}"]
        else:
            # Validate second input file
            second_path = self.validate_input_file(value_or_file, "second input image file")
            operations = [f"-add {second_path}"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def subtract(self,
                input_file: Union[str, Path],
                value_or_file: Union[float, str, Path],
                output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Subtract value or image from input
        
        Args:
            input_file: Input image file path
            value_or_file: Value to subtract or image file path
            output_file: Output image file path
            
        Returns:
            Command execution result dictionary
        """
        if isinstance(value_or_file, (int, float)):
            operations = [f"-sub {value_or_file}"]
        else:
            # Validate second input file
            second_path = self.validate_input_file(value_or_file, "second input image file")
            operations = [f"-sub {second_path}"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def multiply(self,
                input_file: Union[str, Path],
                value_or_file: Union[float, str, Path],
                output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Multiply input by value or image
        
        Args:
            input_file: Input image file path
            value_or_file: Value to multiply by or image file path
            output_file: Output image file path
            
        Returns:
            Command execution result dictionary
        """
        if isinstance(value_or_file, (int, float)):
            operations = [f"-mul {value_or_file}"]
        else:
            # Validate second input file
            second_path = self.validate_input_file(value_or_file, "second input image file")
            operations = [f"-mul {second_path}"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def divide(self,
              input_file: Union[str, Path],
              value_or_file: Union[float, str, Path],
              output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Divide input by value or image
        
        Args:
            input_file: Input image file path
            value_or_file: Value to divide by or image file path
            output_file: Output image file path
            
        Returns:
            Command execution result dictionary
        """
        if isinstance(value_or_file, (int, float)):
            operations = [f"-div {value_or_file}"]
        else:
            # Validate second input file
            second_path = self.validate_input_file(value_or_file, "second input image file")
            operations = [f"-div {second_path}"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def binarize(self,
                input_file: Union[str, Path],
                output_file: Optional[Union[str, Path]] = None,
                threshold: float = 0.5) -> Dict[str, Any]:
        """
        Binarize image
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            threshold: Threshold value for binarization
            
        Returns:
            Command execution result dictionary
        """
        operations = [f"-thr {threshold}", "-bin"]
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def get_help_text(self) -> str:
        """
        Get FSLMaths help information
        
        Returns:
            FSLMaths help text
        """
        return """FSLMaths (Mathematical Operations Tool) Help:

Main Operations:
  -add <value/file> : Add value or image
  -sub <value/file> : Subtract value or image
  -mul <value/file> : Multiply by value or image
  -div <value/file> : Divide by value or image
  -thr <value> : Threshold (lower)
  -uthr <value> : Threshold (upper)
  -thrP <value> : Robust threshold (lower)
  -uthrP <value> : Robust threshold (upper)
  -bin : Binarize
  -s <sigma> : Gaussian smoothing
  -kernel <type> <size> : Apply kernel (mean, median)
  -mas <mask> : Apply mask
  -abs : Absolute value
  -sqrt : Square root
  -log : Natural logarithm
  -exp : Exponential
  -recip : Reciprocal
  -Tmean : Mean across time
  -Tstd : Standard deviation across time
  -Tmax : Maximum across time
  -Tmin : Minimum across time

Usage Examples:
  fslmaths input.nii.gz -thr 0.5 -bin output.nii.gz
  fslmaths input.nii.gz -s 2.0 -mas mask.nii.gz output.nii.gz
  fslmaths input1.nii.gz -add input2.nii.gz -div 2 output.nii.gz
""" 