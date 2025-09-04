"""
FSL FLIRT (Linear Registration Tool) Module

This module provides a Python wrapper for FSL FLIRT tool using fslpy as backend.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging
from .exceptions import FSLParameterError, FSLFileError, FSLCommandError

import fsl.wrappers as fslw
from fsl.wrappers import flirt

from .core import BaseFSLCommand


logger = logging.getLogger(__name__)


class FLIRT(BaseFSLCommand):
    """
    FSL FLIRT (Linear Registration Tool) Wrapper
    
    FLIRT is used for linear registration between images.
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize FLIRT tool
        
        Args:
            timeout: Command execution timeout in seconds
        """
        super().__init__("flirt", timeout=timeout)
    
    def run(self,
            input_file: Union[str, Path],
            reference_file: Union[str, Path],
            output_file: Optional[Union[str, Path]] = None,
            output_matrix: Optional[Union[str, Path]] = None,
            cost_function: str = "mutualinfo",
            search_range: Optional[Tuple[float, float]] = None,
            search_angles: Optional[Tuple[float, float]] = None,
            dof: int = 12,
            interp: str = "trilinear",
            datatype: Optional[str] = None,
            verbose: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Run FLIRT registration
        
        Args:
            input_file: Input image file path
            reference_file: Reference image file path
            output_file: Output image file path
            output_matrix: Output transformation matrix file path
            cost_function: Cost function (mutualinfo, corratio, normcorr, normmi, leastsq, labeldiff, bbr)
            search_range: Search range in mm (min, max)
            search_angles: Search angles in degrees (min, max)
            dof: Degrees of freedom (6, 7, 9, 12)
            interp: Interpolation method (trilinear, nearestneighbour, sinc, spline)
            datatype: Output datatype
            verbose: Verbose output
            **kwargs: Additional FLIRT parameters
            
        Returns:
            Command execution result dictionary
            
        Raises:
            FSLParameterError: When parameters are invalid
            FSLFileError: When file operations fail
        """
        # Validate input files
        input_path = self.validate_input_file(input_file, "input image file")
        reference_path = self.validate_input_file(reference_file, "reference image file")
        
        # Set output file path
        if output_file is None:
            # Handle .nii.gz files properly
            if input_path.name.endswith('.nii.gz'):
                base_name = input_path.name[:-7]
                extension = '.nii.gz'
            elif input_path.name.endswith('.nii'):
                base_name = input_path.name[:-4]
                extension = '.nii'
            else:
                base_name = input_path.stem
                extension = input_path.suffix
            
            output_path = input_path.parent / f"{base_name}_registered{extension}"
        else:
            output_path = self.validate_output_file(output_file)
        
        if output_matrix is None:
            if output_path.name.endswith('.nii.gz'):
                matrix_base = output_path.name[:-7]
            elif output_path.name.endswith('.nii'):
                matrix_base = output_path.name[:-4]
            else:
                matrix_base = output_path.stem
            
            matrix_path = output_path.parent / f"{matrix_base}_matrix.mat"
        else:
            matrix_path = self.validate_output_file(output_matrix)
        
        # Validate parameters
        valid_cost_functions = ["mutualinfo", "corratio", "normcorr", "normmi", "leastsq", "labeldiff", "bbr"]
        if cost_function not in valid_cost_functions:
            raise FSLParameterError(
                f"Invalid cost function: {cost_function}. Must be one of {valid_cost_functions}",
                tool_name="flirt",
                parameter="cost_function"
            )
        
        valid_dofs = [6, 7, 9, 12]
        if dof not in valid_dofs:
            raise FSLParameterError(
                f"Invalid DOF: {dof}. Must be one of {valid_dofs}",
                tool_name="flirt",
                parameter="dof"
            )
        
        valid_interps = ["trilinear", "nearestneighbour", "sinc", "spline"]
        if interp not in valid_interps:
            raise FSLParameterError(
                f"Invalid interpolation: {interp}. Must be one of {valid_interps}",
                tool_name="flirt",
                parameter="interp"
            )
        
        # Build fslpy parameters
        fsl_params = {
            'out': str(output_path),
            'omat': str(matrix_path),
            'cost': cost_function,
            'dof': dof,
            'interp': interp,
            'verbose': verbose
        }
        
        # Add optional parameters
        if search_range:
            fsl_params['searchrx'] = search_range[0]
            fsl_params['searchry'] = search_range[1]
            fsl_params['searchrz'] = search_range[1]
        
        if search_angles:
            fsl_params['searchrx'] = search_angles[0]
            fsl_params['searchry'] = search_angles[1]
            fsl_params['searchrz'] = search_angles[1]
        
        if datatype:
            fsl_params['datatype'] = datatype
        
        # Add additional parameters
        fsl_params.update(kwargs)
        
        # Execute command using fslpy
        logger.info(f"Starting FLIRT registration: {input_path} -> {reference_path}")
        
        try:
            result = flirt(str(input_path), str(reference_path), **fsl_params)
            
            return {
                'success': True,
                'return_code': 0,
                'stdout': str(result),
                'stderr': '',
                'command': f"flirt {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}",
                'output_file': str(output_path),
                'output_matrix': str(matrix_path),
                'input_file': str(input_path),
                'reference_file': str(reference_path)
            }
            
        except Exception as e:
            logger.error(f"FLIRT registration failed: {e}")
            raise FSLCommandError(
                f"FLIRT registration failed: {str(e)}",
                tool_name="flirt",
                command=f"flirt {' '.join([f'{k}={v}' for k, v in fsl_params.items()])}"
            )
    
    def register_affine(self,
                       input_file: Union[str, Path],
                       reference_file: Union[str, Path],
                       output_file: Optional[Union[str, Path]] = None,
                       cost_function: str = "mutualinfo",
                       dof: int = 12) -> Dict[str, Any]:
        """
        Simplified affine registration method
        
        Args:
            input_file: Input image file path
            reference_file: Reference image file path
            output_file: Output image file path
            cost_function: Cost function
            dof: Degrees of freedom
            
        Returns:
            Command execution result dictionary
        """
        return self.run(
            input_file=input_file,
            reference_file=reference_file,
            output_file=output_file,
            cost_function=cost_function,
            dof=dof
        )
    
    def register_rigid(self,
                      input_file: Union[str, Path],
                      reference_file: Union[str, Path],
                      output_file: Optional[Union[str, Path]] = None,
                      cost_function: str = "mutualinfo") -> Dict[str, Any]:
        """
        Rigid body registration method
        
        Args:
            input_file: Input image file path
            reference_file: Reference image file path
            output_file: Output image file path
            cost_function: Cost function
            
        Returns:
            Command execution result dictionary
        """
        return self.run(
            input_file=input_file,
            reference_file=reference_file,
            output_file=output_file,
            cost_function=cost_function,
            dof=6
        )
    
    def get_help_text(self) -> str:
        """
        Get FLIRT help information
        
        Returns:
            FLIRT help text
        """
        return """FLIRT (Linear Registration Tool) Help:

Main Parameters:
  -in <input> : Input image
  -ref <reference> : Reference image
  -out <output> : Output image
  -omat <matrix> : Output transformation matrix
  -cost <cost> : Cost function (mutualinfo, corratio, normcorr, normmi, leastsq, labeldiff, bbr)
  -dof <dof> : Degrees of freedom (6, 7, 9, 12)
  -interp <method> : Interpolation method (trilinear, nearestneighbour, sinc, spline)
  -searchrx <min> <max> : Search range in x direction
  -searchry <min> <max> : Search range in y direction
  -searchrz <min> <max> : Search range in z direction
  -verbose : Verbose output

Usage Examples:
  flirt -in input.nii.gz -ref reference.nii.gz -out registered.nii.gz -omat transform.mat
  flirt -in input.nii.gz -ref reference.nii.gz -out registered.nii.gz -cost mutualinfo -dof 6
""" 