#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import cli

if __name__ == "__main__":
    # Check if the first argument is a command
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            sys.argv.pop(1)  # Remove the 'single' argument
            cli(mode="single")
            sys.exit(0)
        elif sys.argv[1] == "aws":
            sys.argv.pop(1)  # Remove the 'aws' argument
            cli(mode="aws")
            sys.exit(0)
            
    # If no command specified, check if we're in single mode
    # (by looking for a non-flag argument after removing command if present)
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    if args and not any(arg.startswith('--') for arg in sys.argv[1:]):
        cli(mode="single")
    else:
        cli(mode="batch")