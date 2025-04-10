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
        sys.argv.pop(1)  # Remove the 'single' argument
        cli(mode="single")
    else:
        # Check if we're in single mode by looking for a prompt argument
        # (not starting with --) after removing any --model, --schema, or --pydantic args
        args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
        if args and not any(arg.startswith('--') for arg in sys.argv[1:]):
            cli(mode="single")
        else:
            cli(mode="batch")