#!/usr/bin/env python3
"""
Unsloth Multi-GPU support package setup script
"""

import os

from setuptools import find_packages, setup


def read_requirements():
    """Read requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

def read_readme():
    """Read README.md file"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="unsloth-multigpu",
    version="1.0.0",
    description="Unsloth Multi-GPU parallel training support package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Unsloth MultiGPU Team",
    author_email="tanghui315@126.com",
    url="https://github.com/tanghui315/unsloth-multigpu",
    
    packages=find_packages(include=["unsloth_multigpu", "unsloth_multigpu.*"]),
    python_requires=">=3.8",
    
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "datasets>=2.0.0",
        "psutil>=5.8.0",
        "PyYAML>=6.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
    
    extras_require={
        "full": [
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "unsloth-multigpu-test=unsloth_multigpu.tests.run_all_tests:main",
            "unsloth-multigpu-config=unsloth_multigpu.utils.config_utils:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    keywords=[
        "machine-learning",
        "deep-learning", 
        "transformers",
        "multi-gpu",
        "parallel-training",
        "unsloth",
        "fine-tuning",
        "language-models",
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/tanghui315/unsloth-multigpu/issues",
        "Source": "https://github.com/tanghui315/unsloth-multigpu",
        "Documentation": "https://unsloth-multigpu.readthedocs.io/",
    },
    
    include_package_data=True,
    zip_safe=False,
) 