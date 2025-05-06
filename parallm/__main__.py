#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import cli

def main():
    # Typer's cli() will automatically use sys.argv.
    # No need to inspect sys.argv[0] or sys.argv[1] as done previously.
    # The previous logic for distinguishing `python -m` vs direct script,
    # and for popping argv or passing strings like "gemini" to cli(), has been removed
    # as it can conflict with Typer's own argument processing.
    cli()

if __name__ == "__main__":
    main()