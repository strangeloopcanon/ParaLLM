#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import cli

if __name__ == "__main__":
    # If first argument is 'single', remove it and run in single mode
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        sys.argv.pop(1)
        cli(mode="single")
    else:
        cli(mode="batch")