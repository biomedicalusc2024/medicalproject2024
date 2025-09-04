"""
Tests for FSL Wrapper core module

This module contains unit tests for the core functionality of the FSL Wrapper package.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from fslwrapper.core import BaseFSLCommand, FSLCommandResult, FSLPipeline
from fslwrapper.exceptions import FSLNotFoundError, FSLCommandError


class TestFSLCommandResult(unittest.TestCase):
    """Test cases for FSLCommandResult class"""
    
    def test_successful_result(self):
        """Test successful command result"""
        result = FSLCommandResult(
            success=True,
            return_code=0,
            stdout="Success output",
            stderr="",
            command="test_command",
            output_files=["output1.nii.gz", "output2.nii.gz"]
        )
        
        self.assertTrue(result.is_success())
        self.assertEqual(result.get_error_message(), "")
        self.assertEqual(result.return_code, 0)
        self.assertEqual(len(result.output_files), 2)
    
    def test_failed_result(self):
        """Test failed command result"""
        result = FSLCommandResult(
            success=False,
            return_code=1,
            stdout="",
            stderr="Error occurred",
            command="test_command"
        )
        
        self.assertFalse(result.is_success())
        self.assertEqual(result.get_error_message(), "Error occurred")
        self.assertEqual(result.return_code, 1)
    
    def test_string_representation(self):
        """Test string representation of result"""
        result = FSLCommandResult(success=True, return_code=0)
        self.assertIn("Success", str(result))
        
        result = FSLCommandResult(success=False, return_code=1)
        self.assertIn("Failed", str(result))


class TestFSLPipeline(unittest.TestCase):
    """Test cases for FSLPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = FSLPipeline("Test Pipeline")
        self.mock_command = Mock(spec=BaseFSLCommand)
        self.mock_command.run.return_value = {
            'success': True,
            'return_code': 0,
            'stdout': 'Success',
            'stderr': '',
            'command': 'test_command'
        }
    
    def test_add_step(self):
        """Test adding steps to pipeline"""
        self.pipeline.add_step("Step 1", self.mock_command, {'param': 'value'})
        
        self.assertEqual(len(self.pipeline.steps), 1)
        self.assertEqual(self.pipeline.steps[0]['name'], "Step 1")
        self.assertEqual(self.pipeline.steps[0]['command'], self.mock_command)
        self.assertEqual(self.pipeline.steps[0]['parameters'], {'param': 'value'})
    
    def test_chained_add_step(self):
        """Test chained step addition"""
        pipeline = (self.pipeline
                   .add_step("Step 1", self.mock_command, {'param1': 'value1'})
                   .add_step("Step 2", self.mock_command, {'param2': 'value2'}))
        
        self.assertEqual(len(pipeline.steps), 2)
        self.assertEqual(self.pipeline.steps[0]['name'], "Step 1")
        self.assertEqual(self.pipeline.steps[1]['name'], "Step 2")
    
    def test_execute_pipeline_success(self):
        """Test successful pipeline execution"""
        self.pipeline.add_step("Step 1", self.mock_command, {'param': 'value'})
        results = self.pipeline.execute()
        
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_success())
        self.mock_command.run.assert_called_once_with(param='value')
    
    def test_execute_pipeline_failure(self):
        """Test pipeline execution with failure"""
        failing_command = Mock(spec=BaseFSLCommand)
        failing_command.run.return_value = {
            'success': False,
            'return_code': 1,
            'stdout': '',
            'stderr': 'Error occurred',
            'command': 'failing_command'
        }
        
        self.pipeline.add_step("Step 1", self.mock_command, {'param': 'value'})
        self.pipeline.add_step("Step 2", failing_command, {'param': 'value'})
        
        results = self.pipeline.execute(stop_on_error=True)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].is_success())
        self.assertFalse(results[1].is_success())
        self.assertEqual(failing_command.run.call_count, 1)
    
    def test_get_failed_steps(self):
        """Test getting failed steps"""
        success_command = Mock(spec=BaseFSLCommand)
        success_command.run.return_value = {
            'success': True,
            'return_code': 0,
            'stdout': 'Success',
            'stderr': '',
            'command': 'success_command'
        }
        
        failing_command = Mock(spec=BaseFSLCommand)
        failing_command.run.return_value = {
            'success': False,
            'return_code': 1,
            'stdout': '',
            'stderr': 'Error occurred',
            'command': 'failing_command'
        }
        
        self.pipeline.add_step("Step 1", success_command, {})
        self.pipeline.add_step("Step 2", failing_command, {})
        self.pipeline.add_step("Step 3", success_command, {})
        
        self.pipeline.execute(stop_on_error=False)
        failed_steps = self.pipeline.get_failed_steps()
        
        self.assertEqual(len(failed_steps), 1)
        self.assertEqual(failed_steps[0]['step_name'], "Step 2")
        self.assertEqual(failed_steps[0]['step_index'], 1)


class TestBaseFSLCommand(unittest.TestCase):
    """Test cases for BaseFSLCommand abstract class"""
    
    def test_abstract_class_instantiation(self):
        """Test that BaseFSLCommand cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            BaseFSLCommand("test_command")
    
    def test_abstract_method_implementation(self):
        """Test that subclasses must implement run method"""
        class InvalidCommand(BaseFSLCommand):
            pass
        
        with self.assertRaises(TypeError):
            InvalidCommand("test_command")


if __name__ == '__main__':
    unittest.main() 