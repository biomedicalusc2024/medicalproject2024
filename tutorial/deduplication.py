"""
Medical Data Deduplication Tutorial

This script demonstrates how to use the deduplication modules in the medical project 
to remove duplicate entries from biomedical and QA datasets.

The deduplication module provides two main functionalities:
1. QA Deduplication: Removes duplicates from question-answering datasets
2. Biomedical Deduplication: Removes duplicates from biomedical text datasets

Author: Medical Project 2024 Team
Date: September 2025
"""

import os
import sys
import warnings
import argparse
import logging
from pathlib import Path
from typing import List, Optional

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

def setup_environment() -> None:
    """Setup the Python environment and paths"""
    # Get the current script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Calculate project root (assuming this script is in tutorial/ directory)
    project_root = script_dir.parent.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Python path updated with: {project_root}")

def import_deduplication_modules():
    """Import the deduplication modules with error handling"""
    try:
        from medicalproject2024.preprocess.deduplication import deduplicate_qa, deduplicate_biomed
        print("Successfully imported deduplication modules")
        return deduplicate_qa, deduplicate_biomed
    except ImportError as e:
        print(f"Failed to import deduplication modules: {e}")
        print("Please ensure the medicalproject2024 package is properly installed")
        sys.exit(1)

def deduplicate_qa_datasets(datasets: List[str], input_dir: str, output_dir: str, test: bool = True) -> dict:
    """
    Perform QA dataset deduplication
    
    Args:
        datasets: List of QA dataset names to process
        input_dir: Input data directory
        output_dir: Output directory for deduplicated datasets
        test: Whether to run in test mode
        
    Returns:
        dict: Results summary with success/failure information
    """
    print("=" * 60)
    print("QA DATASET DEDUPLICATION")
    if test:
        print("(RUNNING IN TEST MODE)")
    print("=" * 60)
    
    deduplicate_qa, _ = import_deduplication_modules()
    
    # Available QA datasets
    available_qa_datasets = [
        "LiveQA",
        "MedicationQA", 
        "MedMCQA",
        "MedQA-USMLE",
        "PubMedQA"
    ]
    
    # Validate requested datasets
    invalid_datasets = [d for d in datasets if d not in available_qa_datasets]
    if invalid_datasets:
        print(f"Invalid QA datasets requested: {invalid_datasets}")
        print(f"Available QA datasets: {available_qa_datasets}")
    
    valid_datasets = [d for d in datasets if d in available_qa_datasets]
    if not valid_datasets:
        print("No valid QA datasets to process")
        return {"success": [], "failed": datasets}
    
    print(f"Processing {len(valid_datasets)} QA datasets: {valid_datasets}")
    deduplicate_qa(valid_datasets, input_dir, output_dir, test=test)


def deduplicate_biomedical_datasets(datasets: List[str], input_dir: str, output_dir: str, test: bool = True) -> dict:
    """
    Perform biomedical dataset deduplication
    
    Args:
        datasets: List of biomedical dataset names to process
        input_dir: Input data directory
        output_dir: Output directory for deduplicated datasets
        test: Whether to run in test mode
        
    Returns:
        dict: Results summary with success/failure information
    """
    print("=" * 60)
    print("BIOMEDICAL DATASET DEDUPLICATION")
    if test:
        print("(RUNNING IN TEST MODE)")
    print("=" * 60)
    
    _, deduplicate_biomed = import_deduplication_modules()
    
    # Available biomedical datasets
    available_biomed_datasets = [
        "bc5cdr",
        "BioNLI", 
        "CORD19",
        "hoc",
        "SourceData"
    ]
    
    # Validate requested datasets
    invalid_datasets = [d for d in datasets if d not in available_biomed_datasets]
    if invalid_datasets:
        print(f"Invalid biomedical datasets requested: {invalid_datasets}")
        print(f"Available biomedical datasets: {available_biomed_datasets}")
    
    valid_datasets = [d for d in datasets if d in available_biomed_datasets]
    if not valid_datasets:
        print("No valid biomedical datasets to process")
        return {"success": [], "failed": datasets}
    
    print(f"Processing {len(valid_datasets)} biomedical datasets: {valid_datasets}")
    deduplicate_biomed(valid_datasets, input_dir, output_dir, test=test)
    

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Medical Data Deduplication Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific QA datasets
  python deduplication.py --qa-datasets LiveQA MedicationQA --input-dir data --output-dir data

  # Process specific biomedical datasets  
  python deduplication.py --biomed-datasets bc5cdr BioNLI --input-dir data --output-dir data

  # Process both types of datasets
  python deduplication.py --qa-datasets LiveQA --biomed-datasets bc5cdr --input-dir data --output-dir data

  # Quick test with QA datasets
  python deduplication.py --mode qa --qa-datasets LiveQA MedicationQA --input-dir data --output-dir data --test
        """
    )

    parser.add_argument(
        "--mode",
        choices=["qa", "biomed"],
        default="qa",
        help="Processing mode (default: qa)"
    )

    parser.add_argument(
        "--qa-datasets", 
        nargs="*", 
        default=[],
        choices=["LiveQA", "MedicationQA", "MedMCQA", "MedQA-USMLE", "PubMedQA"],
        help="QA datasets to process"
    )
    
    parser.add_argument(
        "--biomed-datasets", 
        nargs="*", 
        default=[],
        choices=["bc5cdr", "BioNLI", "CORD19", "hoc", "SourceData"],
        help="Biomedical datasets to process"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="data",
        help="Input data directory (default: data)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data",
        help="Output directory for deduplicated datasets (default: data)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited data"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "qa":
        # QA 模式：只处理 QA 数据集
        qa_datasets = args.qa_datasets or ["LiveQA", "MedicationQA"]
        biomed_datasets = []
    elif args.mode == "biomed":
        # Biomedical 模式：只处理生物医学数据集
        qa_datasets = []
        biomed_datasets = args.biomed_datasets or ["bc5cdr", "BioNLI"]

    if args.mode == "qa" and not qa_datasets:
        print("No QA datasets specified for QA mode")
        sys.exit(1)
    elif args.mode == "biomed" and not biomed_datasets:
        print("No biomedical datasets specified for biomedical mode")
        sys.exit(1)
    
    # Process QA datasets
    if qa_datasets and args.mode == "qa":
        deduplicate_qa_datasets(qa_datasets, args.input_dir, args.output_dir, args.test)
    
    # Process biomedical datasets
    if biomed_datasets and args.mode == "biomed":
        deduplicate_biomedical_datasets(biomed_datasets, args.input_dir, args.output_dir, args.test)
    

if __name__ == "__main__":
    main()