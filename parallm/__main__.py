#!/usr/bin/env python3
"""
Main entry point for direct module execution.
This allows running the package with 'python -m parallm <args>'
"""
import sys
from parallm.cli import main, query

if __name__ == "__main__":
    # Determine which command to run based on first argument
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Remove the 'query' argument to match the expected format
        sys.argv.pop(1)
        query()
    else:
        main()