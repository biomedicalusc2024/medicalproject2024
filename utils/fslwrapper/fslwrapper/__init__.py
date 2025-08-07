"""
FSL Wrapper - A Python wrapper for FSL tools using fslpy

This package provides a modular and extensible Python interface to FSL (FMRIB Software Library)
tools using fslpy as the backend. It includes wrappers for BET, FLIRT, FAST, and FSLMaths.

Author: 
"""

__version__ = "1.0.0"
__author__ = ""
__email__ = ""

# Import core components
from .core import BaseFSLCommand, FSLCommandResult, FSLPipeline
from .exceptions import (
    FSLWrapperError,
    FSLNotFoundError,
    FSLCommandError,
    FSLParameterError,
    FSLFileError
)

# Import tool modules
from .bet import BET
from .flirt import FLIRT
from .fast import FAST
from .math import FSLMaths

# Import utility functions
from .utils import (
    check_fsl_environment,
    validate_file_path,
    validate_output_path,
    check_fsl_command,
    get_fsl_command_path,
    build_command_args,
    run_fsl_command,
    sanitize_filename,
    get_file_extension,
    is_nifti_file,
    format_file_size
)


class FSLWrapper:
    """
    Unified FSL Wrapper class
    
    This class provides a unified interface to all FSL tools in the package.
    It organizes and provides access to all tools in a convenient way.
    """
    
    def __init__(self, timeout: int = 300):
        """
        Initialize FSL Wrapper
        
        Args:
            timeout: Default timeout for all commands in seconds
        """
        self.timeout = timeout
        
        # Initialize all tools
        self.bet = BET(timeout=timeout)
        self.flirt = FLIRT(timeout=timeout)
        self.fast = FAST(timeout=timeout)
        self.math = FSLMaths(timeout=timeout)
        
        # Store tool references
        self.tools = {
            'bet': self.bet,
            'flirt': self.flirt,
            'fast': self.fast,
            'math': self.math
        }
    
    def get_tool(self, tool_name: str) -> BaseFSLCommand:
        """
        Get a specific tool by name
        
        Args:
            tool_name: Name of the tool (bet, flirt, fast, math)
            
        Returns:
            The requested tool instance
            
        Raises:
            ValueError: If tool name is not recognized
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}")
        
        return self.tools[tool_name]
    
    def list_tools(self) -> list:
        """
        Get list of available tools
        
        Returns:
            List of available tool names
        """
        return list(self.tools.keys())
    
    def check_environment(self) -> dict:
        """
        Check FSL environment configuration
        
        Returns:
            Dictionary containing FSL environment information
        """
        return check_fsl_environment()
    
    def create_pipeline(self, name: str = "FSL Pipeline") -> FSLPipeline:
        """
        Create a new FSL processing pipeline
        
        Args:
            name: Name of the pipeline
            
        Returns:
            FSLPipeline instance
        """
        return FSLPipeline(name=name)
    
    def run_brain_extraction(self,
                           input_file: str,
                           output_file: str = None,
                           fractional_intensity: float = 0.5,
                           robust: bool = True) -> dict:
        """
        Run brain extraction using BET
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            fractional_intensity: Fractional intensity threshold
            robust: Use robust center estimation
            
        Returns:
            Command execution result
        """
        return self.bet.extract_brain(
            input_file=input_file,
            output_file=output_file,
            fractional_intensity=fractional_intensity,
            robust=robust
        )
    
    def run_registration(self,
                        input_file: str,
                        reference_file: str,
                        output_file: str = None,
                        cost_function: str = "mutualinfo",
                        dof: int = 12) -> dict:
        """
        Run image registration using FLIRT
        
        Args:
            input_file: Input image file path
            reference_file: Reference image file path
            output_file: Output image file path
            cost_function: Cost function for registration
            dof: Degrees of freedom
            
        Returns:
            Command execution result
        """
        return self.flirt.register_affine(
            input_file=input_file,
            reference_file=reference_file,
            output_file=output_file,
            cost_function=cost_function,
            dof=dof
        )
    
    def run_segmentation(self,
                        input_file: str,
                        output_basename: str = None,
                        number_classes: int = 3,
                        bias_correction: bool = True) -> dict:
        """
        Run tissue segmentation using FAST
        
        Args:
            input_file: Input image file path
            output_basename: Output basename for segmentation files
            number_classes: Number of tissue classes
            bias_correction: Apply bias field correction
            
        Returns:
            Command execution result
        """
        return self.fast.segment_t1(
            input_file=input_file,
            output_basename=output_basename,
            number_classes=number_classes,
            bias_correction=bias_correction
        )
    
    def run_math_operation(self,
                          input_file: str,
                          output_file: str = None,
                          operations: list = None) -> dict:
        """
        Run mathematical operations using FSLMaths
        
        Args:
            input_file: Input image file path
            output_file: Output image file path
            operations: List of mathematical operations
            
        Returns:
            Command execution result
        """
        return self.math.run(
            input_file=input_file,
            output_file=output_file,
            operations=operations
        )
    
    def get_version_info(self) -> dict:
        """
        Get version information for all tools
        
        Returns:
            Dictionary containing version information
        """
        version_info = {
            'package_version': __version__,
            'tools': {}
        }
        
        for tool_name, tool in self.tools.items():
            try:
                version_info['tools'][tool_name] = tool.get_version()
            except Exception as e:
                version_info['tools'][tool_name] = f"Error: {str(e)}"
        
        return version_info
    
    def __str__(self) -> str:
        """Return string representation of FSL Wrapper"""
        return f"FSLWrapper(timeout={self.timeout}, tools={list(self.tools.keys())})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of FSL Wrapper"""
        return (f"FSLWrapper("
                f"timeout={self.timeout}, "
                f"tools={list(self.tools.keys())}, "
                f"version={__version__})")


# Convenience function to create FSL Wrapper instance
def create_fsl_wrapper(timeout: int = 300) -> FSLWrapper:
    """
    Create a new FSL Wrapper instance
    
    Args:
        timeout: Default timeout for all commands in seconds
        
    Returns:
        FSLWrapper instance
    """
    return FSLWrapper(timeout=timeout)


# Export main classes and functions
__all__ = [
    # Main wrapper class
    'FSLWrapper',
    'create_fsl_wrapper',
    
    # Core classes
    'BaseFSLCommand',
    'FSLCommandResult',
    'FSLPipeline',
    
    # Tool classes
    'BET',
    'FLIRT',
    'FAST',
    'FSLMaths',
    
    # Exception classes
    'FSLWrapperError',
    'FSLNotFoundError',
    'FSLCommandError',
    'FSLParameterError',
    'FSLFileError',
    
    # Utility functions
    'check_fsl_environment',
    'validate_file_path',
    'validate_output_path',
    'check_fsl_command',
    'get_fsl_command_path',
    'build_command_args',
    'run_fsl_command',
    'sanitize_filename',
    'get_file_extension',
    'is_nifti_file',
    'format_file_size',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
] 