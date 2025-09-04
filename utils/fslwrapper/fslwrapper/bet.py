"""
FSL BET (Brain Extraction Tool) Module

This module provides a Python wrapper for FSL BET tool using fslpy as backend.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from .exceptions import FSLParameterError, FSLFileError, FSLCommandError

import fsl.wrappers as fslw
from fsl.wrappers import bet

from .core import BaseFSLCommand


logger = logging.getLogger(__name__)


class BET(BaseFSLCommand):
    """
    FSL BET (Brain Extraction Tool) Wrapper
    
    BET is used to extract brain tissue from T1-weighted images, removing skull and other non-brain tissue.
    """
    
    def __init__(self, timeout: int = 600):
        """
        Initialize BET tool
        
        Args:
            timeout: Command execution timeout in seconds, BET typically takes longer
        """
        super().__init__("bet", timeout=timeout)
    
    def run(self, 
            input_file: Union[str, Path],
            output_file: Optional[Union[str, Path]] = None,
            fractional_intensity: Optional[float] = None,
            vertical_gradient: Optional[float] = None,
            center_of_gravity: Optional[tuple] = None,
            radius: Optional[float] = None,
            threshold: bool = False,
            robust: bool = False,
            padding: bool = False,
            remove_eyes: bool = False,
            reduce_bias: bool = False,
            surface: bool = False,
            mesh: bool = False,
            outline: bool = False,
            skull: bool = False,
            head_radius: Optional[float] = None,
            no_output: bool = False,
            mask: bool = False,
            mesh_fmesh: bool = False,
            mesh_vmesh: bool = False,
            mesh_surfaces: bool = False,
            mesh_volumes: bool = False,
            verbose: bool = False) -> Dict[str, Any]:
        """
        Run BET brain extraction
        
        Args:
            input_file: Input image file path
            output_file: Output image file path (optional, default adds _brain suffix)
            fractional_intensity: Fractional intensity threshold (0-1)
            vertical_gradient: Vertical gradient fraction (0-1)
            center_of_gravity: Center of gravity coordinates (x, y, z)
            radius: Head radius in mm
            threshold: Use thresholding
            robust: Use robust center estimation
            padding: Use padding
            remove_eyes: Remove eyes
            reduce_bias: Reduce bias field
            surface: Generate surface
            mesh: Generate mesh
            outline: Generate outline
            skull: Generate skull
            head_radius: Head radius in mm
            no_output: No output
            mask: Generate mask
            mesh_fmesh: Generate fast mesh
            mesh_vmesh: Generate detailed mesh
            mesh_surfaces: Generate mesh surfaces
            mesh_volumes: Generate mesh volumes
            verbose: Verbose output
            
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
            # Handle .nii.gz files properly
            if input_path.name.endswith('.nii.gz'):
                base_name = input_path.name[:-7]  # Remove .nii.gz
                extension = '.nii.gz'
            elif input_path.name.endswith('.nii'):
                base_name = input_path.name[:-4]  # Remove .nii
                extension = '.nii'
            else:
                base_name = input_path.stem
                extension = input_path.suffix
            
            if base_name.endswith('_brain'):
                output_path = input_path.parent / f"{base_name}_extracted{extension}"
            else:
                output_path = input_path.parent / f"{base_name}_brain{extension}"
        else:
            output_path = self.validate_output_file(output_file)
        
        # Build fslpy parameters
        fsl_params = {}
        
        # Add optional parameters
        if fractional_intensity is not None:
            if not 0 <= fractional_intensity <= 1:
                raise FSLParameterError(
                    "fractional_intensity must be between 0-1",
                    tool_name="bet",
                    parameter="fractional_intensity"
                )
            fsl_params['f'] = fractional_intensity
        
        if vertical_gradient is not None:
            if not 0 <= vertical_gradient <= 1:
                raise FSLParameterError(
                    "vertical_gradient must be between 0-1",
                    tool_name="bet",
                    parameter="vertical_gradient"
                )
            fsl_params['g'] = vertical_gradient
        
        if center_of_gravity is not None:
            if len(center_of_gravity) != 3:
                raise FSLParameterError(
                    "center_of_gravity must be a 3-element tuple",
                    tool_name="bet",
                    parameter="center_of_gravity"
                )
            fsl_params['c'] = center_of_gravity
        
        if radius is not None:
            if radius <= 0:
                raise FSLParameterError(
                    "radius must be greater than 0",
                    tool_name="bet",
                    parameter="radius"
                )
            fsl_params['r'] = radius
        
        if head_radius is not None:
            if head_radius <= 0:
                raise FSLParameterError(
                    "head_radius must be greater than 0",
                    tool_name="bet",
                    parameter="head_radius"
                )
            fsl_params['head_radius'] = head_radius
        
        # Add boolean parameters
        if threshold:
            fsl_params['t'] = True
        if robust:
            fsl_params['R'] = True
        if padding:
            fsl_params['p'] = True
        if remove_eyes:
            fsl_params['e'] = True
        if reduce_bias:
            fsl_params['B'] = True
        if surface:
            fsl_params['s'] = True
        if mesh:
            fsl_params['m'] = True
        if outline:
            fsl_params['o'] = True
        if skull:
            fsl_params['S'] = True
        if no_output:
            fsl_params['n'] = True
        if mask:
            fsl_params['mask'] = True
        if mesh_fmesh:
            fsl_params['mesh_fmesh'] = True
        if mesh_vmesh:
            fsl_params['mesh_vmesh'] = True
        if mesh_surfaces:
            fsl_params['mesh_surfaces'] = True
        if mesh_volumes:
            fsl_params['mesh_volumes'] = True
        if verbose:
            fsl_params['verbose'] = True
        
        # Execute command using fslpy
        logger.info(f"Starting BET brain extraction: {input_path} -> {output_path}")
        
        try:
            result = bet(str(input_path), str(output_path), **fsl_params)
            
            return {
                'success': True,
                'return_code': 0,
                'stdout': str(result),
                'stderr': '',
                'command': f"bet {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}",
                'output_file': str(output_path),
                'input_file': str(input_path)
            }
            
        except Exception as e:
            logger.error(f"BET brain extraction failed: {e}")
            raise FSLCommandError(
                f"BET brain extraction failed: {str(e)}",
                tool_name="bet",
                command=f"bet {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}"
            )
    
    def extract_brain(self, 
                     input_file: Union[str, Path],
                     output_file: Optional[Union[str, Path]] = None,
                     fractional_intensity: float = 0.5,
                     robust: bool = True,
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Simplified brain extraction method
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            fractional_intensity: Fractional intensity threshold (default 0.5)
            robust: Use robust center estimation (default True)
            verbose: Verbose output
            
        Returns:
            Command execution result dictionary
        """
        return self.run(
            input_file=input_file,
            output_file=output_file,
            fractional_intensity=fractional_intensity,
            robust=robust,
            verbose=verbose
        )
    
    def extract_brain_with_mask(self,
                               input_file: Union[str, Path],
                               output_file: Optional[Union[str, Path]] = None,
                               mask_file: Optional[Union[str, Path]] = None,
                               fractional_intensity: float = 0.5,
                               robust: bool = True) -> Dict[str, Any]:
        """
        Brain extraction method with mask
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            mask_file: Mask file path (optional)
            fractional_intensity: Fractional intensity threshold
            robust: Use robust center estimation
            
        Returns:
            Command execution result dictionary
        """
        # Set mask file path
        if mask_file is None:
            input_path = Path(input_file)
            mask_file = input_path.parent / f"{input_path.stem}_mask{input_path.suffix}"
        
        return self.run(
            input_file=input_file,
            output_file=output_file,
            fractional_intensity=fractional_intensity,
            robust=robust,
            mask=True
        )
    
    def get_help_text(self) -> str:
        """
        Get BET help information
        
        Returns:
            BET help text
        """
        return """BET (Brain Extraction Tool) Help:

Main Parameters:
  -f <f> : Fractional intensity threshold (0-1, default: 0.5)
  -g <g> : Vertical gradient fraction (0-1, default: 0)
  -c <x> <y> <z> : Center of gravity coordinates
  -r <r> : Head radius (mm)
  -R : Robust center estimation
  -S : Generate skull image
  -B : Reduce bias field
  -e : Remove eyes
  -p : Use padding
  -t : Thresholding
  -s : Generate surface
  -m : Generate mesh
  -o : Generate outline
  -mask : Generate mask
  -v : Verbose output

Usage Examples:
  bet input.nii.gz output_brain.nii.gz -f 0.3 -R
  bet input.nii.gz output_brain.nii.gz -c 90 100 130 -r 75
""" 