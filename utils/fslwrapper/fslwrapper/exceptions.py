"""
FSL Wrapper Custom Exception Class Definitions

This module defines all custom exception classes used in the FSL wrapper library.
"""


class FSLWrapperError(Exception):
    """
    Base exception class for FSL wrapper
    
    All FSL wrapper related exceptions should inherit from this class.
    """
    
    def __init__(self, message: str, tool_name: str = None, command: str = None):
        """
        Initialize FSL wrapper exception
        
        Args:
            message: Error message
            tool_name: FSL tool name that caused the error
            command: Executed command
        """
        self.message = message
        self.tool_name = tool_name
        self.command = command
        
        # Build complete error message
        full_message = message
        if tool_name:
            full_message = f"[{tool_name}] {full_message}"
        if command:
            full_message = f"{full_message} (Command: {command})"
            
        super().__init__(full_message)


class FSLNotFoundError(FSLWrapperError):
    """
    FSL environment not found exception
    
    Raised when FSL environment is not properly configured or FSLDIR
    environment variable is not set.
    """
    
    def __init__(self, message: str = "FSL environment not found"):
        """
        Initialize FSL not found exception
        
        Args:
            message: Error message
        """
        super().__init__(message, tool_name="FSL")


class FSLCommandError(FSLWrapperError):
    """
    FSL command execution error exception
    
    Raised when FSL command execution fails.
    """
    
    def __init__(self, message: str, tool_name: str, command: str, return_code: int = None):
        """
        Initialize FSL command error exception
        
        Args:
            message: Error message
            tool_name: FSL tool name that caused the error
            command: Executed command
            return_code: Command return code
        """
        self.return_code = return_code
        super().__init__(message, tool_name, command)


class FSLParameterError(FSLWrapperError):
    """
    FSL parameter error exception
    
    Raised when parameters passed to FSL command are invalid.
    """
    
    def __init__(self, message: str, tool_name: str, parameter: str = None):
        """
        Initialize FSL parameter error exception
        
        Args:
            message: Error message
            tool_name: FSL tool name that caused the error
            parameter: Problematic parameter name
        """
        self.parameter = parameter
        super().__init__(message, tool_name)


class FSLFileError(FSLWrapperError):
    """
    FSL file operation error exception
    
    Raised when file operations fail (e.g., file not found, insufficient permissions, etc.).
    """
    
    def __init__(self, message: str, tool_name: str, file_path: str = None):
        """
        Initialize FSL file error exception
        
        Args:
            message: Error message
            tool_name: FSL tool name that caused the error
            file_path: Problematic file path
        """
        self.file_path = file_path
        super().__init__(message, tool_name) 