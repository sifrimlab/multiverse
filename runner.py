#!/usr/bin/env python3
"""
Entry point script for running the multiverse workflow.

This script provides a command-line interface for executing the multiverse
multimodal data integration workflow.
"""

import sys
import os
import argparse


def main():
    """Main entry point for the runner script."""
    parser = argparse.ArgumentParser(
        description="Run multiverse multimodal data integration workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py config.json          # Run with custom config
  python runner.py                      # Run with default config (config_alldatasets.json)
        """
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config_alldatasets.json",
        help="Path to the JSON configuration file (default: config_alldatasets.json)"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if configuration file exists
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {args.config_path}\n"
                f"Please provide a valid configuration file path or ensure "
                f"'{args.config_path}' exists in the current directory."
            )
        
        print(f"Starting workflow with config: {args.config_path}")
        
        # Import here to allow --help to work without dependencies installed
        from multiverse.main import main_workflow
        
        # Execute the main workflow
        main_workflow(args.config_path)
        
        print("Workflow completed successfully")
        
    except FileNotFoundError as e:
        print(f"CRITICAL EXECUTION ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"CRITICAL EXECUTION ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
