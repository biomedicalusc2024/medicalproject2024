"""
Tests for FSL Wrapper BET module

This module contains unit tests for the BET (Brain Extraction Tool) functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from fslwrapper.bet import BET
from fslwrapper.exceptions import FSLParameterError, FSLFileError, FSLNotFoundError


class TestBET(unittest.TestCase):
    """Test cases for BET class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('fslwrapper.core.check_fsl_environment') as mock_check:
            mock_check.return_value = {'FSLDIR': '/usr/local/fsl', 'version': '6.0.0'}
            with patch('fslwrapper.core.check_fsl_command') as mock_cmd:
                mock_cmd.return_value = True
                with patch('fslwrapper.core.get_fsl_command_path') as mock_path:
                    mock_path.return_value = '/usr/local/fsl/bin/bet'
                    self.bet = BET()
    
    def test_initialization(self):
        """Test BET initialization"""
        self.assertEqual(self.bet.command_name, "bet")
        self.assertEqual(self.bet.timeout, 600)
        self.assertIsNotNone(self.bet.fsl_info)
        self.assertIsNotNone(self.bet.command_path)
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.bet.bet')
    def test_run_basic(self, mock_bet, mock_validate_output, mock_validate_input):
        """Test basic BET run"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        mock_validate_output.return_value = Path("/tmp/input_brain.nii.gz")
        
        mock_bet.return_value = "bet result"
        
        result = self.bet.run(
            input_file="/tmp/input.nii.gz",
            output_file="/tmp/output.nii.gz"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['return_code'], 0)
        mock_bet.assert_called_once()
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.bet.bet')
    def test_run_with_parameters(self, mock_bet, mock_validate_output, mock_validate_input):
        """Test BET run with parameters"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        mock_bet.return_value = "bet result"
        
        result = self.bet.run(
            input_file="/tmp/input.nii.gz",
            output_file="/tmp/output.nii.gz",
            fractional_intensity=0.3,
            robust=True,
            verbose=True
        )
        
        self.assertTrue(result['success'])
        mock_bet.assert_called_once()
        
        call_args = mock_bet.call_args[1]
        self.assertEqual(call_args['f'], 0.3)
        self.assertTrue(call_args['R'])
        self.assertTrue(call_args['verbose'])
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_fractional_intensity(self, mock_validate_input):
        """Test invalid fractional intensity parameter"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        
        with self.assertRaises(FSLParameterError):
            self.bet.run(
                input_file="/tmp/input.nii.gz",
                fractional_intensity=1.5  # Invalid value > 1
            )
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_vertical_gradient(self, mock_validate_input):
        """Test invalid vertical gradient parameter"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        
        with self.assertRaises(FSLParameterError):
            self.bet.run(
                input_file="/tmp/input.nii.gz",
                vertical_gradient=-0.1  # Invalid value < 0
            )
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_center_of_gravity(self, mock_validate_input):
        """Test invalid center of gravity parameter"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        
        with self.assertRaises(FSLParameterError):
            self.bet.run(
                input_file="/tmp/input.nii.gz",
                center_of_gravity=(1, 2)  # Invalid: not 3 elements
            )
    
    @patch('fslwrapper.core.validate_file_path')
    def test_invalid_radius(self, mock_validate_input):
        """Test invalid radius parameter"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        
        with self.assertRaises(FSLParameterError):
            self.bet.run(
                input_file="/tmp/input.nii.gz",
                radius=0  # Invalid: not > 0
            )
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.bet.bet')
    def test_extract_brain_method(self, mock_bet, mock_validate_output, mock_validate_input):
        """Test extract_brain convenience method"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        mock_bet.return_value = "bet result"
        
        result = self.bet.extract_brain(
            input_file="/tmp/input.nii.gz",
            output_file="/tmp/output.nii.gz",
            fractional_intensity=0.4,
            robust=True
        )
        
        self.assertTrue(result['success'])
        mock_bet.assert_called_once()
        
        call_args = mock_bet.call_args[1]
        self.assertEqual(call_args['f'], 0.4)
        self.assertTrue(call_args['R'])
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.bet.bet')
    def test_extract_brain_with_mask_method(self, mock_bet, mock_validate_output, mock_validate_input):
        """Test extract_brain_with_mask convenience method"""
        # Mock file validation
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        # Mock fslpy bet function
        mock_bet.return_value = "bet result"
        
        # Test extract_brain_with_mask method
        result = self.bet.extract_brain_with_mask(
            input_file="/tmp/input.nii.gz",
            output_file="/tmp/output.nii.gz",
            fractional_intensity=0.5,
            robust=True
        )
        
        self.assertTrue(result['success'])
        mock_bet.assert_called_once()
        
        call_args = mock_bet.call_args[1]
        self.assertTrue(call_args['mask'])
    
    def test_get_help_text(self):
        """Test get_help_text method"""
        help_text = self.bet.get_help_text()
        self.assertIsInstance(help_text, str)
        self.assertIn("BET", help_text)
        self.assertIn("Parameters", help_text)
        self.assertIn("Examples", help_text)
    
    @patch('fslwrapper.core.validate_file_path')
    @patch('fslwrapper.core.validate_output_path')
    @patch('fslwrapper.bet.bet')
    def test_bet_failure_handling(self, mock_bet, mock_validate_output, mock_validate_input):
        """Test BET failure handling"""
        mock_validate_input.return_value = Path("/tmp/input.nii.gz")
        mock_validate_output.return_value = Path("/tmp/output.nii.gz")
        
        mock_bet.side_effect = Exception("BET failed")
        
        with self.assertRaises(Exception):
            self.bet.run(
                input_file="/tmp/input.nii.gz",
                output_file="/tmp/output.nii.gz"
            )


if __name__ == '__main__':
    unittest.main() 