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
        
    Returns:
        bool: True if test completed successfully
    """
    print("=" * 60)
    print("LITS (LIVER TUMOR SEGMENTATION) TEST")
    if test_mode:
        print("(RUNNING IN TEST MODE)")
    print("=" * 60)
    
    # Print detailed explanation of what this test does
    print("\n TEST OVERVIEW:")
    print("This test evaluates a quality control model for liver segmentation masks.")
    print("The model predicts Dice coefficients to assess segmentation quality.")
    print("\n WHAT IS DICE COEFFICIENT?")
    print("- Dice coefficient measures overlap between predicted and ground truth masks")
    print("- Range: 0.0 (no overlap) to 1.0 (perfect overlap)")
    print("- Values > 0.7 typically indicate good segmentation quality")
    print("- Values < 0.5 may indicate poor segmentation quality")
    
    try:
        from medicalproject2024.preprocess.Imaging_3D.quality_control.mask.test_lits import default_lits
        print("\n Successfully imported LiTS test module")
        
        # Run the test
        print("\n Running LiTS test...")
        print("-" * 30)
        
        try:
            default_lits()
            print("-" * 30)
            
            # Enhanced results explanation
            print("\n UNDERSTANDING THE RESULTS:")
            print("=" * 40)
            print("• 'Average predicted Dice': Mean quality score across all test samples")
            print("• 'Detailed Results': Individual quality predictions for each CT slice")
            print("\n INTERPRETATION GUIDE:")
            print("• Dice > 0.8: Excellent segmentation quality")
            print("• Dice 0.6-0.8: Good segmentation quality") 
            print("• Dice 0.4-0.6: Moderate segmentation quality")
            print("• Dice < 0.4: Poor segmentation quality")
            print("\n TECHNICAL DETAILS:")
            print("• Model: Pre-trained neural network for quality assessment")
            print("• Input: CT images and corresponding segmentation masks")
            print("• Output: Predicted Dice coefficient for each mask")
            print("• Dataset: LiTS (Liver Tumor Segmentation Challenge)")
            
            print("\n LiTS test completed successfully")
            return True
            
        except Exception as e:
            print("-" * 30)
            print(f" LiTS test failed: {e}")
            
            # Enhanced error handling with more specific guidance
            print("\n TROUBLESHOOTING GUIDE:")
            if "Bad CRC-32" in str(e) or "corrupted" in str(e).lower():
                print(" CORRUPTED DOWNLOAD DETECTED:")
                print("   • Clear download cache: rm -rf ./data/LiTS")
                print("   • Check available disk space (need ~100MB)")
                print("   • Ensure stable internet connection for download")
                print("   • Try running the test again")
            elif "No module named" in str(e):
                print(" MISSING DEPENDENCY DETECTED:")
                print("   • Install required packages:")
                print("     pip install torch torchvision gdown tqdm matplotlib")
                print("   • Verify all Imaging_3D modules are present")
                print("   • Check Python path configuration")
            elif "CUDA" in str(e) or "device" in str(e).lower():
                print("  GPU/DEVICE ISSUE DETECTED:")
                print("   • Check PyTorch installation: python -c 'import torch; print(torch.__version__)'")
                print("   • Verify CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'")
                print("   • Consider running on CPU if GPU issues persist")
            elif "download" in str(e).lower() or "url" in str(e).lower():
                print(" DOWNLOAD ISSUE DETECTED:")
                print("   • Check internet connection")
                print("   • Verify Google Drive access (dataset hosted there)")
                print("   • Try again later if server is temporarily unavailable")
            else:
                print(" GENERAL ERROR:")
                print(f"   • Error details: {str(e)}")
                print("   • Check log files for more information")
                print("   • Ensure all dependencies are properly installed")
            
            return False
        
    except ImportError as e:
        print(f" Failed to import LiTS test module: {e}")
        print("\n MODULE IMPORT TROUBLESHOOTING:")
        print("• Ensure medicalproject2024 package is properly installed")
        print("• Check if all Imaging_3D modules are in correct directory")
        print("• Verify Python path includes project root")
        print("• Try reinstalling the package dependencies")
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
    
    print("\n 3D Medical Imaging Processing Tutorial")
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