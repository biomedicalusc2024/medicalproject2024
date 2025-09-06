"""
3D Medical Imaging Processing Tutorial

This script demonstrates how to use the Imaging_3D modules in the medical project 
to process 3D medical images with quality control, preprocessing, and segmentation capabilities.

The Imaging_3D module provides functionality for:
1. Image Quality Control: Automated quality assessment for 3D medical images
2. Mask Quality Control: Quality assessment and validation for segmentation masks
3. 3D Image Preprocessing: Standardization and preprocessing of 3D medical images
4. TotalSegmentator Integration: Advanced anatomical segmentation using TotalSegmentator

Author: Medical Project 2024 Team
Date: September 2025
"""

import os
import sys
import warnings
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )

def setup_environment() -> Path:
    """Setup the Python environment and paths"""
    # Get the current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Calculate project root (assuming this script is in tutorial/ directory)
    project_root = script_dir.parent.parent
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path updated with: {project_root}")
    
    return project_root

def check_dependencies() -> Dict[str, bool]:
    """Check for required dependencies for 3D imaging"""
    dependencies = {
        "torch": "PyTorch for deep learning",
        "torchvision": "PyTorch vision utilities", 
        "nibabel": "Neuroimaging data I/O",
        "SimpleITK": "Medical image processing",
        "matplotlib": "Plotting and visualization",
        "pandas": "Data manipulation",
        "numpy": "Numerical computing",
        "tqdm": "Progress bars",
        "pickle": "Data serialization"
    }
    
    print("Dependency Check:")
    print("-" * 50)
    
    missing_deps = []
    available_deps = {}
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"{dep:15}: {description}")
            available_deps[dep] = True
        except ImportError:
            print(f"{dep:15}: {description} (MISSING)")
            missing_deps.append(dep)
            available_deps[dep] = False
    
    if missing_deps:
        print(f"\nMissing dependencies: {missing_deps}")
        print("Install with: pip install " + " ".join(missing_deps))
    else:
        print("\nAll dependencies available!")
    
    return available_deps

def import_imaging3d_modules():
    """Import the Imaging_3D modules with error handling"""
    import_results = {}
    
    modules_to_test = [
        ("medicalproject2024.preprocess.Imaging_3D.quality_control.mask.dataset", "Mask Dataset"),
        ("medicalproject2024.preprocess.Imaging_3D.quality_control.mask.model", "Mask Model"),
        ("medicalproject2024.preprocess.Imaging_3D.quality_control.mask.test_lits", "LiTS Test"),
    ]
    
    print("\nImporting Imaging_3D Modules:")
    print("-" * 50)
    
    for module_path, module_name in modules_to_test:
        try:
            __import__(module_path)
            import_results[module_name] = True
            print(f"{module_name:20}: Successfully imported")
        except ImportError as e:
            import_results[module_name] = False
            print(f"{module_name:20}: Import failed - {str(e)[:50]}...")
        except Exception as e:
            import_results[module_name] = False
            print(f"{module_name:20}: Warning - {str(e)[:50]}...")
    
    # Check TotalSegmentator files without importing (due to import issues)
    try:
        project_root = setup_environment()
        imaging_3d_path = project_root / "medicalproject2024" / "preprocess" / "Imaging_3D"
        inference_file = imaging_3d_path / "quality_control" / "mask" / "inference_TotalSegmentator.py"
        model_file = imaging_3d_path / "quality_control" / "mask" / "model.py"
        
        if inference_file.exists() and model_file.exists():
            import_results["TotalSegmentator"] = True
            print(f"{'TotalSegmentator':20}: Files available")
        else:
            import_results["TotalSegmentator"] = False
            print(f"{'TotalSegmentator':20}: Required files missing")
    except Exception as e:
        import_results["TotalSegmentator"] = False
        print(f"{'TotalSegmentator':20}: Check failed - {str(e)[:50]}...")
    
    return import_results

def run_lits_test(test_mode: bool = True) -> bool:
    """
    Run LiTS (Liver Tumor Segmentation) test
    
    Args:
        test_mode: Whether to run in test mode
        output_dir: Output directory for results
        
    Returns:
        bool: True if test completed successfully
    """
    print("=" * 60)
    print("LITS (LIVER TUMOR SEGMENTATION) TEST")
    if test_mode:
        print("(RUNNING IN TEST MODE)")
    print("=" * 60)
    
    try:
        from medicalproject2024.preprocess.Imaging_3D.quality_control.mask.test_lits import default_lits
        print("Successfully imported LiTS test module")
        
        
        # Run the test
        print("\nRunning LiTS test...")
        print("-" * 30)
        
        try:
            default_lits()
            print("-" * 30)
            print("LiTS test completed successfully")
            return True
            
        except Exception as e:
            print("-" * 30)
            print(f"LiTS test failed: {e}")
            
            # Provide specific guidance based on error type
            if "Bad CRC-32" in str(e) or "corrupted" in str(e).lower():
                print("\nCorrupted download detected. Suggested fixes:")
                print("   1. Clear download cache and retry")
                print("   2. Check available disk space")
                print("   3. Try downloading with stable internet connection")
            elif "No module named" in str(e):
                print("\nMissing dependency detected. Suggested fixes:")
                print("   1. Install missing packages: pip install torch torchvision gdown tqdm")
                print("   2. Check if all required files are present")
            elif "CUDA" in str(e) or "device" in str(e).lower():
                print("\nGPU/Device issue detected. Suggested fixes:")
                print("   1. Ensure PyTorch is properly installed")
                print("   2. Check if CUDA is available (if using GPU)")
                print("   3. Try running on CPU instead")
            
            return False
        
    except ImportError as e:
        print(f"Failed to import LiTS test module: {e}")
        print("Please ensure all Imaging_3D modules are properly installed")
        return False


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="3D Medical Imaging Processing Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LiTS test (default mode)
  python Imaging_3d.py
        """
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Setup environment
    project_root = setup_environment()
    
    print("\n🏥 3D Medical Imaging Processing Tutorial")
    print("=" * 60)
    
    # Check dependencies
    deps_available = check_dependencies()
    
    # Import modules
    import_results = import_imaging3d_modules()
    
    # Check if basic requirements are met
    required_modules = ["Mask Dataset", "Mask Model", "LiTS Test"]
    modules_available = all(import_results.get(mod, False) for mod in required_modules)
    
    if not modules_available:
        print("\nSome required modules are not available.")
        print("Please check the installation and try again.")
    
    # Execute based on mode
    success = True
        
    success = run_lits_test()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    successful_imports = sum(1 for available in import_results.values() if available)
    total_imports = len(import_results)
    
    print(f"Module Imports: {successful_imports}/{total_imports} successful")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()