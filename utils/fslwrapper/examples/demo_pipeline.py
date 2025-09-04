#!/usr/bin/env python3
"""
FSL Wrapper Demo Pipeline

Demonstrates a typical neuroimaging processing pipeline using the FSL Wrapper.
Performs brain extraction, registration, and tissue segmentation.
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fslwrapper import FSLWrapper, create_fsl_wrapper
from fslwrapper.exceptions import FSLWrapperError, FSLNotFoundError


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fsl_pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_environment(fsl_wrapper):
    """Check FSL environment and print information"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check FSL environment
        fsl_info = fsl_wrapper.check_environment()
        logger.info("FSL Environment Information:")
        for key, value in fsl_info.items():
            logger.info(f"  {key}: {value}")
        
        # Get version information
        version_info = fsl_wrapper.get_version_info()
        logger.info("Version Information:")
        logger.info(f"  Package Version: {version_info['package_version']}")
        for tool, version in version_info['tools'].items():
            logger.info(f"  {tool}: {version}")
        
        return True
        
    except FSLNotFoundError as e:
        logger.error(f"FSL Environment Error: {e}")
        return False


def create_sample_data():
    """Create sample data for demonstration"""
    logger = logging.getLogger(__name__)
    
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    fsl_standard_dir = Path("/Users/bourneli/fsl/data/standard")
    
    mni152_2mm = fsl_standard_dir / "MNI152_T1_2mm.nii.gz"
    mni152_1mm = fsl_standard_dir / "MNI152_T1_1mm.nii.gz"
    
    reference_file = None
    if mni152_2mm.exists():
        reference_file = mni152_2mm
        logger.info(f"Found MNI152 2mm template: {reference_file}")
    elif mni152_1mm.exists():
        reference_file = mni152_1mm
        logger.info(f"Found MNI152 1mm template: {reference_file}")
    else:
        logger.warning("MNI152 template not found in FSL standard data")
        logger.info("You can download it using: fsl-get_standard MNI152_T1_2mm")
    
    input_file = data_dir / "demo_input.nii.gz"
    
    if not input_file.exists():
        if reference_file:
            import shutil
            shutil.copy2(reference_file, input_file)
            logger.info(f"Created demo input file from template: {input_file}")
        else:
            logger.warning("No input file available and no template found")
            logger.info("Please provide a T1-weighted image for processing")
            return None, None
    
    return input_file, reference_file


def run_brain_extraction_pipeline(fsl_wrapper, input_file):
    """Run brain extraction pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STEP 1: Brain Extraction")
    logger.info("=" * 50)
    
    try:
        # Run brain extraction (let BET generate the output filename automatically)
        result = fsl_wrapper.run_brain_extraction(
            input_file=str(input_file),
            output_file=None,  # Let BET handle the filename
            fractional_intensity=0.5,
            robust=True
        )
        
        if result['success']:
            logger.info("Brain extraction completed successfully!")
            logger.info(f"Output file: {result['output_file']}")
            return result['output_file']
        else:
            logger.error("Brain extraction failed!")
            return None
            
    except Exception as e:
        logger.error(f"Brain extraction error: {e}")
        return None


def run_registration_pipeline(fsl_wrapper, input_file, reference_file):
    """Run registration pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STEP 2: Image Registration")
    logger.info("=" * 50)
    
    try:
        result = fsl_wrapper.run_registration(
            input_file=str(input_file),
            reference_file=str(reference_file),
            output_file=None,
            cost_function="mutualinfo",
            dof=12
        )
        
        if result['success']:
            logger.info("Registration completed successfully!")
            logger.info(f"Output file: {result['output_file']}")
            logger.info(f"Transformation matrix: {result['output_matrix']}")
            return result['output_file']
        else:
            logger.error("Registration failed!")
            return None
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return None


def run_segmentation_pipeline(fsl_wrapper, input_file):
    """Run tissue segmentation pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STEP 3: Tissue Segmentation")
    logger.info("=" * 50)
    
    try:
        # Run segmentation
        input_path = Path(input_file)
        result = fsl_wrapper.run_segmentation(
            input_file=str(input_file),
            output_basename=str(input_path.parent / f"{input_path.stem}_seg"),
            number_classes=3,
            bias_correction=True
        )
        
        if result['success']:
            logger.info("Segmentation completed successfully!")
            logger.info(f"Output files: {result['output_files']}")
            return result['output_files']
        else:
            logger.error("Segmentation failed!")
            return None
            
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return None


def run_post_processing_pipeline(fsl_wrapper, segmentation_files):
    """Run post-processing pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("STEP 4: Post-Processing")
    logger.info("=" * 50)
    
    try:
        # Get the segmentation file (safely handle the list)
        if not segmentation_files or len(segmentation_files) < 2:
            logger.warning("Not enough segmentation files for post-processing")
            return None
            
        seg_file = segmentation_files[1]  # _seg.nii.gz file
        
        # Create binary mask from segmentation
        result = fsl_wrapper.run_math_operation(
            input_file=seg_file,
            output_file=str(Path(seg_file).parent / "brain_mask.nii.gz"),
            operations=["-thr", "1", "-bin"]
        )
        
        if result['success']:
            logger.info("Post-processing completed successfully!")
            logger.info(f"Brain mask: {result['output_file']}")
            return result['output_file']
        else:
            logger.error("Post-processing failed!")
            return None
            
    except Exception as e:
        logger.error(f"Post-processing error: {e}")
        return None


def create_processing_pipeline(fsl_wrapper, input_file, reference_file):
    """Create and run a complete processing pipeline"""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating FSL processing pipeline...")
    
    # Create pipeline
    pipeline = fsl_wrapper.create_pipeline("Complete FSL Processing Pipeline")
    
    # Calculate the correct filenames first
    if input_file.name.endswith('.nii.gz'):
        base_name = input_file.name[:-7]
        extension = '.nii.gz'
    elif input_file.name.endswith('.nii'):
        base_name = input_file.name[:-4]
        extension = '.nii'
    else:
        base_name = input_file.stem
        extension = input_file.suffix
    
    bet_output = input_file.parent / f"{base_name}_brain{extension}"
    
    # Add brain extraction step
    pipeline.add_step(
        "Brain Extraction",
        fsl_wrapper.bet,
        {
            'input_file': str(input_file),
            'output_file': None,
            'fractional_intensity': 0.5,
            'robust': True
        }
    )
    
    # Add registration step (if reference file is available)
    if reference_file:
        
        pipeline.add_step(
            "Image Registration",
            fsl_wrapper.flirt,
            {
                'input_file': str(bet_output),
                'reference_file': str(reference_file),
                'output_file': None,  # Let FLIRT handle the filename
                'cost_function': "mutualinfo",
                'dof': 12
            }
        )
    
    # Add segmentation step
    if reference_file:
        # Use the registered brain image
        flirt_output = input_file.parent / f"{base_name}_brain_registered{extension}"
        seg_input = flirt_output
        seg_basename = input_file.parent / f"{base_name}_brain_registered_seg"
    else:
        # Use the brain extracted image
        seg_input = bet_output
        seg_basename = input_file.parent / f"{base_name}_brain_seg"
    
    pipeline.add_step(
        "Tissue Segmentation",
        fsl_wrapper.fast,
        {
            'input_file': str(seg_input),
            'output_basename': None,  # Let FAST handle the filename
            'number_classes': 3,
            'bias_correction': True
        }
    )
    
    return pipeline


def main():
    """Main function to run the demo pipeline"""
    logger = setup_logging()
    
    logger.info("FSL Wrapper Demo Pipeline")
    logger.info("=" * 50)
    
    try:
        # Create FSL wrapper instance
        fsl_wrapper = create_fsl_wrapper(timeout=600)
        logger.info("FSL Wrapper initialized successfully")
        
        # Check environment
        if not check_environment(fsl_wrapper):
            logger.error("FSL environment check failed. Exiting.")
            return 1
        
        # Get sample data
        input_file, reference_file = create_sample_data()
        if not input_file:
            logger.error("No input file available. Exiting.")
            logger.info("To run this demo, you need:")
            logger.info("1. A T1-weighted image file, or")
            logger.info("2. FSL standard data (MNI152 template)")
            logger.info("You can download FSL standard data using: fsl-get_standard MNI152_T1_2mm")
            return 1
        
        logger.info(f"Input file: {input_file}")
        if reference_file:
            logger.info(f"Reference file: {reference_file}")
        
        logger.info("\n" + "=" * 50)
        logger.info("DEMO OPTIONS:")
        logger.info("1. Run full pipeline (brain extraction + registration + segmentation)")
        logger.info("2. Run individual steps with detailed output")
        logger.info("3. Test environment only (no processing)")
        logger.info("=" * 50)
        
        # For demo purposes, we'll run the full pipeline
        # In a real application, you could add user input here
        choice = 1  # Default to full pipeline
        
        if choice == 3:
            logger.info("Environment test completed successfully!")
            return 0
        
        # Option 1: Run individual steps
        logger.info("\nRunning individual processing steps...")
        
        # Step 1: Brain extraction
        brain_file = run_brain_extraction_pipeline(fsl_wrapper, input_file)
        if not brain_file:
            logger.error("Brain extraction failed. Stopping pipeline.")
            return 1
        
        # Step 2: Registration (if reference file available)
        if reference_file:
            registered_file = run_registration_pipeline(fsl_wrapper, brain_file, reference_file)
            if not registered_file:
                logger.error("Registration failed. Stopping pipeline.")
                return 1
            seg_input_file = registered_file
        else:
            seg_input_file = brain_file
        
        # Step 3: Segmentation
        seg_files = run_segmentation_pipeline(fsl_wrapper, seg_input_file)
        if not seg_files:
            logger.error("Segmentation failed. Stopping pipeline.")
            return 1
        
        # Step 4: Post-processing
        mask_file = run_post_processing_pipeline(fsl_wrapper, seg_files)
        if not mask_file:
            logger.error("Post-processing failed.")
        
        # Option 2: Run as a pipeline
        logger.info("\nRunning as a complete pipeline...")
        pipeline = create_processing_pipeline(fsl_wrapper, input_file, reference_file)
        results = pipeline.execute(stop_on_error=True)
        
        # Check pipeline results
        failed_steps = pipeline.get_failed_steps()
        if failed_steps:
            logger.error(f"Pipeline failed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                logger.error(f"  - {step['step_name']}: {step['result'].get_error_message()}")
        else:
            logger.info("Pipeline completed successfully!")
        
        logger.info("\nDemo pipeline completed!")
        logger.info("Check the output files in the sample_data directory.")
        
        return 0
        
    except FSLWrapperError as e:
        logger.error(f"FSL Wrapper Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 