"""Setup configuration for the multiverse package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="multiverse",
    version="0.1.0",
    description="A package for comparing MOFA, MOWGLI, MultiVI, and PCA on multimodal datasets",
    author="Multi-verse Contributors",
    author_email="",
    url="https://github.com/sifrimlab/multi-verse",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            # Canonical CLI entry point
            "multiverse=multiverse.cli_entrypoints:main",
            # Backward-compatible CLI entry point for docker-based workflow
            "multiverse-cli=multiverse.runner.cli:main",
            # Maintenance utility: rebuilds run_metrics table from artifacts
            "multiverse-rebuild-metrics=multiverse.tools.rebuild_run_metrics:main",
        ],
    },
)
