"""
Tests for FSL Wrapper FLIRT module

This module contains unit tests for the FLIRT (Linear Registration Tool) functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from fslwrapper.flirt import FLIRT
from fslwrapper.exceptions import FSLParameterError, FSLFileError, FSLNotFoundError


class TestFLIRT(unittest.TestCase):
    """Test cases for FLIRT class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock FSL environment check
        with patch('fslwrapper.core.check_fsl_environment') as mock_check:
            mock_check.return_value = {'FSLDIR': '/usr/local/fsl', 'version': '6.0.0'}
            with patch('fslwrapper.core.check_fsl_command') as mock_cmd:
                mock_cmd.return_value = True
                with patch('fslwrapper.core.get_fsl_command_path') as mock_path:
                    mock_path.return_value = '/usr/local/fsl/bin/flirt'
                    self.flirt = FLIRT()
    
    def test_initialization(self):
        """Test FLIRT initialization"""
        self.assertEqual(self.flirt.command_name, "flirt")
        self.assertEqual(self.flirt.timeout, 300)
        self.assertIsNotNone(self.flirt.fsl_info)
        self.assertIsNotNone(self.flirt.command_path)
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_run_basic(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test basic FLIRT run"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function
        mock_flirt.return_value = "flirt result"
        
        # Test basic run
        result = self.flirt.run(
            input_file="/tmp/input.nii.gz",
            reference_file="/tmp/reference.nii.gz",
            output_file="/tmp/output.nii.gz"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['return_code'], 0)
        mock_flirt.assert_called_once()
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_run_with_parameters(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test FLIRT run with parameters"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function
        mock_flirt.return_value = "flirt result"
        
        # Test run with parameters
        result = self.flirt.run(
            input_file="/tmp/input.nii.gz",
            reference_file="/tmp/reference.nii.gz",
            output_file="/tmp/output.nii.gz",
            cost_function="corratio",
            dof=6,
            interp="sinc",
            verbose=True
        )
        
        self.assertTrue(result['success'])
        mock_flirt.assert_called_once()
        
        # Check that parameters were passed correctly
        call_args = mock_flirt.call_args[1]
        self.assertEqual(call_args['cost'], "corratio")
        self.assertEqual(call_args['dof'], 6)
        self.assertEqual(call_args['interp'], "sinc")
        self.assertTrue(call_args['verbose'])
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_cost_function(self, mock_validate_input):
        """Test invalid cost function parameter"""
        # Mock file validation to pass
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        
        with self.assertRaises(FSLParameterError):
            self.flirt.run(
                input_file="/tmp/input.nii.gz",
                reference_file="/tmp/reference.nii.gz",
                cost_function="invalid_cost"
            )
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_dof(self, mock_validate_input):
        """Test invalid degrees of freedom parameter"""
        # Mock file validation to pass
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        
        with self.assertRaises(FSLParameterError):
            self.flirt.run(
                input_file="/tmp/input.nii.gz",
                reference_file="/tmp/reference.nii.gz",
                dof=5  # Invalid DOF
            )
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_interpolation(self, mock_validate_input):
        """Test invalid interpolation parameter"""
        # Mock file validation to pass
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        
        with self.assertRaises(FSLParameterError):
            self.flirt.run(
                input_file="/tmp/input.nii.gz",
                reference_file="/tmp/reference.nii.gz",
                interp="invalid_interp"
            )
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_register_affine_method(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test register_affine convenience method"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function
        mock_flirt.return_value = "flirt result"
        
        # Test register_affine method
        result = self.flirt.register_affine(
            input_file="/tmp/input.nii.gz",
            reference_file="/tmp/reference.nii.gz",
            output_file="/tmp/output.nii.gz",
            cost_function="mutualinfo",
            dof=12
        )
        
        self.assertTrue(result['success'])
        mock_flirt.assert_called_once()
        
        # Check that parameters were passed correctly
        call_args = mock_flirt.call_args[1]
        self.assertEqual(call_args['cost'], "mutualinfo")
        self.assertEqual(call_args['dof'], 12)
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_register_rigid_method(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test register_rigid convenience method"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function
        mock_flirt.return_value = "flirt result"
        
        # Test register_rigid method
        result = self.flirt.register_rigid(
            input_file="/tmp/input.nii.gz",
            reference_file="/tmp/reference.nii.gz",
            output_file="/tmp/output.nii.gz",
            cost_function="corratio"
        )
        
        self.assertTrue(result['success'])
        mock_flirt.assert_called_once()
        
        # Check that DOF was set to 6 for rigid registration
        call_args = mock_flirt.call_args[1]
        self.assertEqual(call_args['dof'], 6)
    
    def test_get_help_text(self):
        """Test get_help_text method"""
        help_text = self.flirt.get_help_text()
        self.assertIsInstance(help_text, str)
        self.assertIn("FLIRT", help_text)
        self.assertIn("Parameters", help_text)
        self.assertIn("Examples", help_text)
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_flirt_failure_handling(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test FLIRT failure handling"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function to raise exception
        mock_flirt.side_effect = Exception("FLIRT failed")
        
        # Test that exception is properly handled
        with self.assertRaises(Exception):
            self.flirt.run(
                input_file="/tmp/input.nii.gz",
                reference_file="/tmp/reference.nii.gz",
                output_file="/tmp/output.nii.gz"
            )
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.flirt.flirt')
    def test_search_range_parameters(self, mock_flirt, mock_validate_output, mock_validate_input):
        """Test search range parameters"""
        # Mock file validation
        mock_validate_input.side_effect = [Path("/tmp/input.nii.gz"), Path("/tmp/reference.nii.gz")]
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy flirt function
        mock_flirt.return_value = "flirt result"
        
        # Test with search range
        result = self.flirt.run(
            input_file="/tmp/input.nii.gz",
            reference_file="/tmp/reference.nii.gz",
            search_range=(-10, 10)
        )
        
        self.assertTrue(result['success'])
        mock_flirt.assert_called_once()
        
        # Check that search range parameters were set
        call_args = mock_flirt.call_args[1]
        self.assertEqual(call_args['searchrx'], -10)
        self.assertEqual(call_args['searchry'], 10)
        self.assertEqual(call_args['searchrz'], 10)


if __name__ == '__main__':
    unittest.main() 