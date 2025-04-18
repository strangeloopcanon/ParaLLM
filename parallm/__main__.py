#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import cli

def main():
    # Check if we're running as a module (python -m parallm) or directly (parallm)
    if sys.argv[0].endswith('__main__.py'):
        # Running as module (python -m parallm)
        if len(sys.argv) > 1:
            if sys.argv[1] == "aws":
                cli("aws")
            elif sys.argv[1] == "gemini":
                cli("gemini")
            else:
                cli()
        else:
            cli()
    else:
        # Running directly (parallm)
        if len(sys.argv) > 1:
            if sys.argv[1] == "aws":
                sys.argv.pop(1)  # Remove the mode argument
                cli("aws")
            elif sys.argv[1] == "gemini":
                sys.argv.pop(1)  # Remove the mode argument
                cli("gemini")
            else:
                cli()
        else:
            cli()

if __name__ == "__main__":
    main()