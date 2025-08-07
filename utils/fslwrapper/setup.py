#!/usr/bin/env python3
"""
Setup script for FSL Wrapper package

This script installs the FSL Wrapper package and its dependencies.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith("#") and ";" not in line
        ]
        # Filter out development dependencies
        requirements = [req for req in requirements if not any(
            dev_pkg in req for dev_pkg in ["pytest", "black", "flake8", "mypy", "sphinx"]
        )]

setup(
    name="fslwrapper",
    version="1.0.0",
    author="FSL Wrapper Team",
    author_email="fsl-wrapper@example.com",
    description="A Python wrapper for FSL tools using fslpy as backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/fslwrapper",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/fslwrapper/issues",
        "Source": "https://github.com/your-repo/fslwrapper",
        "Documentation": "https://github.com/your-repo/fslwrapper/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "fslwrapper=fslwrapper.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fslwrapper": [
            "*.py",
            "*.md",
            "*.txt",
        ],
    },
    keywords=[
        "neuroimaging",
        "fsl",
        "brain",
        "mri",
        "medical",
        "neuroscience",
        "bet",
        "flirt",
        "fast",
        "fslmaths",
    ],
    platforms=["Linux", "MacOS"],
    license="MIT",
    zip_safe=False,
) 