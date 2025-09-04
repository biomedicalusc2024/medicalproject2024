"""
FSL Wrapper Command Line Interface

This module provides a command-line interface for the FSL Wrapper package.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from . import FSLWrapper, __version__
from .exceptions import FSLWrapperError


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_info(args) -> int:
    """Show FSL environment information"""
    try:
        wrapper = FSLWrapper()
        env_info = wrapper.check_environment()
        version_info = wrapper.get_version_info()
        
        print("FSL Wrapper Information:")
        print(f"  Package Version: {version_info['package_version']}")
        print("\nFSL Environment:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        
        print("\nTool Versions:")
        for tool, version in version_info['tools'].items():
            print(f"  {tool}: {version}")
        
        return 0
        
    except FSLWrapperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_bet(args) -> int:
    """Run brain extraction"""
    try:
        wrapper = FSLWrapper()
        
        result = wrapper.run_brain_extraction(
            input_file=args.input,
            output_file=args.output,
            fractional_intensity=args.fractional_intensity,
            robust=args.robust
        )
        
        if result['success']:
            print(f"Brain extraction completed: {result['output_file']}")
            return 0
        else:
            print(f"Brain extraction failed: {result.get('stderr', 'Unknown error')}", file=sys.stderr)
            return 1
            
    except FSLWrapperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_flirt(args) -> int:
    """Run image registration"""
    try:
        wrapper = FSLWrapper()
        
        result = wrapper.run_registration(
            input_file=args.input,
            reference_file=args.reference,
            output_file=args.output,
            cost_function=args.cost_function,
            dof=args.dof
        )
        
        if result['success']:
            print(f"Registration completed: {result['output_file']}")
            if 'output_matrix' in result:
                print(f"Transformation matrix: {result['output_matrix']}")
            return 0
        else:
            print(f"Registration failed: {result.get('stderr', 'Unknown error')}", file=sys.stderr)
            return 1
            
    except FSLWrapperError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="FSL Wrapper - Python interface to FSL tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'FSL Wrapper {__version__}'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show FSL environment information')
    info_parser.set_defaults(func=cmd_info)
    
    # BET command
    bet_parser = subparsers.add_parser('bet', help='Brain extraction tool')
    bet_parser.add_argument('input', help='Input image file')
    bet_parser.add_argument('-o', '--output', help='Output image file')
    bet_parser.add_argument('-f', '--fractional-intensity', type=float, default=0.5,
                           help='Fractional intensity threshold (default: 0.5)')
    bet_parser.add_argument('-R', '--robust', action='store_true',
                           help='Use robust center estimation')
    bet_parser.set_defaults(func=cmd_bet)
    
    # FLIRT command
    flirt_parser = subparsers.add_parser('flirt', help='Image registration tool')
    flirt_parser.add_argument('input', help='Input image file')
    flirt_parser.add_argument('reference', help='Reference image file')
    flirt_parser.add_argument('-o', '--output', help='Output image file')
    flirt_parser.add_argument('-cost', '--cost-function', default='mutualinfo',
                             choices=['mutualinfo', 'corratio', 'normcorr', 'normmi', 'leastsq'],
                             help='Cost function (default: mutualinfo)')
    flirt_parser.add_argument('-dof', '--dof', type=int, default=12,
                             choices=[6, 7, 9, 12],
                             help='Degrees of freedom (default: 12)')
    flirt_parser.set_defaults(func=cmd_flirt)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    setup_logging(args.verbose)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())