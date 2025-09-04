"""
FSL Wrapper Utility Functions Module

This module provides various utility functions used in the FSL wrapper library,
including environment checking, path validation, etc.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging

from .exceptions import FSLNotFoundError, FSLFileError, FSLCommandError


logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


def check_fsl_environment() -> Dict[str, str]:
    """
    Check if FSL environment is properly configured
    
    Returns:
        Dictionary containing FSL environment information
        
    Raises:
        FSLNotFoundError: When FSL environment is not found
    """
    fsl_info = {}
    fsldir = os.environ.get('FSLDIR')
    if not fsldir:
        raise FSLNotFoundError("FSLDIR environment variable not set")
    
    fsl_info['FSLDIR'] = fsldir
    
    if not os.path.exists(fsldir):
        raise FSLNotFoundError(f"FSLDIR directory does not exist: {fsldir}")
    
    try:
        result = subprocess.run(['fslversion'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        fsl_info['version'] = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Unable to get FSL version information")
        fsl_info['version'] = "Unknown"
            
    fsl_bin_paths = [
        os.path.join(fsldir, 'bin'),
        os.path.join(fsldir, 'share', 'fsl', 'bin'),
        os.path.join(fsldir, 'sbin')
    ]
    
    path_found = False
    for fsl_bin_path in fsl_bin_paths:
        if fsl_bin_path in os.environ.get('PATH', ''):
            path_found = True
            break
    
    if not path_found:
        logger.warning(f"FSL binary paths not in PATH: {fsl_bin_paths}")
        logger.info("FSL commands will be found via FSLDIR path lookup")
    
    return fsl_info


def validate_file_path(file_path: Union[str, Path], 
                      must_exist: bool = True,
                      file_type: str = "file") -> Path:
    """
    Validate file path validity
    
    Args:
        file_path: File path
        must_exist: Whether file must exist
        file_type: File type description (for error messages)
        
    Returns:
        Validated Path object
        
    Raises:
        FSLFileError: When file path is invalid
    """
    try:
        path = Path(file_path).resolve()
        
        if must_exist and not path.exists():
            raise FSLFileError(
                f"{file_type} does not exist: {path}",
                tool_name="utils",
                file_path=str(path)
            )
        
        return path
    except Exception as e:
        raise FSLFileError(
            f"Invalid {file_type} path: {file_path}",
            tool_name="utils",
            file_path=str(file_path)
        ) from e


def validate_output_path(output_path: Union[str, Path], 
                        create_parent: bool = True) -> Path:
    """
    Validate output path and ensure parent directory exists
    
    Args:
        output_path: Output file path
        create_parent: Whether to create parent directory
        
    Returns:
        Validated Path object
        
    Raises:
        FSLFileError: When output path is invalid
    """
    try:
        path = Path(output_path).resolve()
        parent_dir = path.parent
        
        if create_parent and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {parent_dir}")
        
        return path
    except Exception as e:
        raise FSLFileError(
            f"Invalid output path: {output_path}",
            tool_name="utils",
            file_path=str(output_path)
        ) from e


def check_fsl_command(command: str) -> bool:
    """
    Check if FSL command is available
    
    Args:
        command: FSL command name to check
        
    Returns:
        Whether command is available
    """
    command_path = get_fsl_command_path(command)
    if not command_path:
        return False
    
    if not os.access(command_path, os.X_OK):
        return False
    
    help_options = ['--help', '-help', '-h', '-H']
    
    for help_opt in help_options:
        try:
            result = subprocess.run([command_path, help_opt], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode in [0, 1] and len(result.stdout) > 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    return os.path.exists(command_path) and os.access(command_path, os.X_OK)


def get_fsl_command_path(command: str) -> Optional[str]:
    """
    Get full path of FSL command
    
    Args:
        command: FSL command name
        
    Returns:
        Full path of command, or None if not found
    """
    command_path = shutil.which(command)
    if command_path:
        return command_path
    
    fsldir = os.environ.get('FSLDIR')
    if fsldir:
        possible_paths = [
            os.path.join(fsldir, 'bin', command),
            os.path.join(fsldir, 'share', 'fsl', 'bin', command),
            os.path.join(fsldir, 'sbin', command)
        ]
        
        for fsl_bin_path in possible_paths:
            if os.path.exists(fsl_bin_path):
                return fsl_bin_path
    
    return None


def build_command_args(base_command: str, 
                      parameters: Dict[str, Any]) -> List[str]:
    """
    Build FSL command argument list
    
    Args:
        base_command: Base command
        parameters: Parameter dictionary
        
    Returns:
        Complete command argument list
    """
    args = [base_command]
    
    for key, value in parameters.items():
        if value is None:
            continue
            
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            if key.startswith('--'):
                args.extend([key, str(value)])
            else:
                args.extend([f"--{key}", str(value)])
    
    return args


def run_fsl_command(command_args: List[str], 
                   timeout: int = 300,
                   capture_output: bool = True) -> subprocess.CompletedProcess:
    """
    Execute FSL command
    
    Args:
        command_args: Command argument list
        timeout: Timeout in seconds
        capture_output: Whether to capture output
        
    Returns:
        Command execution result
        
    Raises:
        FSLCommandError: When command execution fails
    """
    try:
        logger.info(f"Executing command: {' '.join(command_args)}")
        
        result = subprocess.run(
            command_args,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False
        )
        
        if result.returncode != 0:
            error_msg = f"Command execution failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nError output: {result.stderr}"
            
            raise FSLCommandError(
                error_msg,
                tool_name=command_args[0],
                command=' '.join(command_args),
                return_code=result.returncode
            )
        
        return result
        
    except subprocess.TimeoutExpired:
        raise FSLCommandError(
            f"Command execution timeout (>{timeout} seconds)",
            tool_name=command_args[0],
            command=' '.join(command_args)
        )
    except FileNotFoundError:
        raise FSLNotFoundError(f"Command not found: {command_args[0]}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing unsafe characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    unsafe_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    filename = filename.strip()
    
    return filename


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension
    
    Args:
        file_path: File path
        
    Returns:
        File extension (including dot)
    """
    return Path(file_path).suffix


def is_nifti_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is in NIfTI format
    
    Args:
        file_path: File path
        
    Returns:
        Whether file is NIfTI format
    """
    path = Path(file_path)
    return path.suffix.lower() in ['.nii', '.nii.gz']


def format_file_size(size_bytes: int) -> str:
    """
    Format file size display
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB" 