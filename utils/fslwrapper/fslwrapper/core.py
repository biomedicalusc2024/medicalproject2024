"""
FSL Wrapper Core Module

This module defines the core abstract base classes for the FSL wrapper library.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

from .exceptions import FSLWrapperError, FSLNotFoundError, FSLCommandError, FSLParameterError, FSLFileError
from .utils import (
    check_fsl_environment, 
    validate_file_path, 
    validate_output_path,
    check_fsl_command,
    get_fsl_command_path,
    build_command_args,
    run_fsl_command
)


logger = logging.getLogger(__name__)


class BaseFSLCommand(ABC):
    """
    Abstract base class for FSL commands
    
    All FSL tools should inherit from this class, providing unified interface
    and common functionality.
    """
    
    def __init__(self, command_name: str, timeout: int = 300):
        """
        Initialize FSL command base class
        
        Args:
            command_name: FSL command name
            timeout: Command execution timeout in seconds
        """
        self.command_name = command_name
        self.timeout = timeout
        self.fsl_info = None
        self.command_path = None
        
        # Check FSL environment
        self._check_environment()
        
        # Validate command availability
        self._validate_command()
    
    def _check_environment(self) -> None:
        """
        Check FSL environment configuration
        
        Raises:
            FSLNotFoundError: When FSL environment is not properly configured
        """
        try:
            self.fsl_info = check_fsl_environment()
            logger.info(f"FSL environment check passed: {self.fsl_info}")
        except FSLNotFoundError as e:
            logger.error(f"FSL environment check failed: {e}")
            raise
    
    def _validate_command(self) -> None:
        """
        Validate if FSL command is available
        
        Raises:
            FSLNotFoundError: When command is not available
        """
        # Get command path
        self.command_path = get_fsl_command_path(self.command_name)
        
        if not self.command_path:
            raise FSLNotFoundError(f"FSL command not found: {self.command_name}")
        
        # Check if command is executable
        if not check_fsl_command(self.command_name):
            raise FSLNotFoundError(f"FSL command not executable: {self.command_name}")
        
        logger.info(f"FSL command validation passed: {self.command_name} -> {self.command_path}")
    
    def validate_input_file(self, file_path: Union[str, Path], 
                           file_type: str = "input file") -> Path:
        """
        Validate input file path
        
        Args:
            file_path: File path
            file_type: File type description
            
        Returns:
            Validated Path object
        """
        return validate_file_path(file_path, must_exist=True, file_type=file_type)
    
    def validate_output_file(self, file_path: Union[str, Path], 
                           create_parent: bool = True) -> Path:
        """
        Validate output file path
        
        Args:
            file_path: File path
            create_parent: Whether to create parent directory
            
        Returns:
            Validated Path object
        """
        return validate_output_path(file_path, create_parent=create_parent)
    
    def execute_command(self, parameters: Dict[str, Any], 
                       input_files: Optional[List[str]] = None,
                       output_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute FSL command
        
        Args:
            parameters: Command parameter dictionary
            input_files: Input file list (for validation)
            output_files: Output file list (for validation)
            
        Returns:
            Command execution result dictionary
            
        Raises:
            FSLCommandError: When command execution fails
        """
        # Validate input files
        if input_files:
            for file_path in input_files:
                self.validate_input_file(file_path)
        
        # Validate output file paths
        if output_files:
            for file_path in output_files:
                self.validate_output_file(file_path)
        
        # Build command arguments
        command_args = build_command_args(self.command_name, parameters)
        
        # Execute command
        try:
            result = run_fsl_command(command_args, timeout=self.timeout)
            
            return {
                'success': True,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(command_args)
            }
            
        except FSLCommandError as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to run FSL command
        
        Subclasses must implement this method to provide specific command
        execution logic.
        
        Args:
            **kwargs: Command parameters
            
        Returns:
            Command execution result
        """
        pass
    
    def get_help(self) -> str:
        """
        Get command help information
        
        Returns:
            Command help text
        """
        try:
            result = run_fsl_command([self.command_name, '--help'], timeout=30)
            return result.stdout
        except FSLCommandError:
            return f"Unable to get help information for {self.command_name}"
    
    def get_version(self) -> str:
        """
        Get command version information
        
        Returns:
            Command version information
        """
        try:
            result = run_fsl_command([self.command_name, '--version'], timeout=30)
            return result.stdout.strip()
        except FSLCommandError:
            return "Version information not available"
    
    def __str__(self) -> str:
        """Return string representation of command"""
        return f"{self.__class__.__name__}(command={self.command_name})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of command"""
        return (f"{self.__class__.__name__}("
                f"command={self.command_name}, "
                f"path={self.command_path}, "
                f"timeout={self.timeout})")


class FSLCommandResult:
    """
    FSL command execution result class
    
    Used to encapsulate and standardize FSL command execution results.
    """
    
    def __init__(self, success: bool, 
                 return_code: int = None,
                 stdout: str = "",
                 stderr: str = "",
                 command: str = "",
                 output_files: List[str] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize command execution result
        
        Args:
            success: Whether command executed successfully
            return_code: Command return code
            stdout: Standard output
            stderr: Standard error
            command: Executed command
            output_files: Output file list
            metadata: Additional metadata
        """
        self.success = success
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.command = command
        self.output_files = output_files or []
        self.metadata = metadata or {}
    
    def is_success(self) -> bool:
        """Check if command executed successfully"""
        return self.success and self.return_code == 0
    
    def get_error_message(self) -> str:
        """Get error message"""
        if self.is_success():
            return ""
        return self.stderr or f"Command execution failed, return code: {self.return_code}"
    
    def __str__(self) -> str:
        """Return string representation of result"""
        status = "Success" if self.is_success() else "Failed"
        return f"FSLCommandResult({status}, return_code={self.return_code})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of result"""
        return (f"FSLCommandResult("
                f"success={self.success}, "
                f"return_code={self.return_code}, "
                f"command='{self.command}', "
                f"output_files={self.output_files})")


class FSLPipeline:
    """
    FSL processing pipeline class
    
    Used to build and execute FSL processing pipelines.
    """
    
    def __init__(self, name: str = "FSL Pipeline"):
        """
        Initialize FSL processing pipeline
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps = []
        self.results = []
    
    def add_step(self, step_name: str, command: BaseFSLCommand, 
                 parameters: Dict[str, Any]) -> 'FSLPipeline':
        """
        Add processing step
        
        Args:
            step_name: Step name
            command: FSL command object
            parameters: Command parameters
            
        Returns:
            Pipeline object (supports chaining)
        """
        self.steps.append({
            'name': step_name,
            'command': command,
            'parameters': parameters
        })
        return self
    
    def execute(self, stop_on_error: bool = True) -> List[FSLCommandResult]:
        """
        Execute processing pipeline
        
        Args:
            stop_on_error: Whether to stop on error
            
        Returns:
            Execution result list
        """
        self.results = []
        
        for i, step in enumerate(self.steps):
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {step['name']}")
            
            try:
                result_dict = step['command'].run(**step['parameters'])
                result = FSLCommandResult(
                    success=result_dict.get('success', False),
                    return_code=result_dict.get('return_code'),
                    stdout=result_dict.get('stdout', ''),
                    stderr=result_dict.get('stderr', ''),
                    command=result_dict.get('command', ''),
                    metadata={'step_name': step['name']}
                )
                
                self.results.append(result)
                
                if not result.is_success() and stop_on_error:
                    logger.error(f"Step failed: {step['name']}")
                    break
                    
            except Exception as e:
                logger.error(f"Step execution exception: {step['name']} - {e}")
                result = FSLCommandResult(
                    success=False,
                    stderr=str(e),
                    metadata={'step_name': step['name'], 'exception': str(e)}
                )
                self.results.append(result)
                
                if stop_on_error:
                    break
        
        return self.results
    
    def get_failed_steps(self) -> List[Dict[str, Any]]:
        """Get failed steps"""
        failed_steps = []
        for i, result in enumerate(self.results):
            if not result.is_success():
                failed_steps.append({
                    'step_index': i,
                    'step_name': self.steps[i]['name'],
                    'result': result
                })
        return failed_steps
    
    def __str__(self) -> str:
        """Return string representation of pipeline"""
        return f"FSLPipeline({self.name}, steps={len(self.steps)})" 