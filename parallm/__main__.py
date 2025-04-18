#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import cli

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "aws":
            cli("aws")
        elif sys.argv[1] == "gemini":
            cli("gemini")
        else:
            # If the first argument is not a known mode, treat it as a prompt
            cli()
    else:
        cli()

if __name__ == "__main__":
    main()