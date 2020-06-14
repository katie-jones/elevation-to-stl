#!/usr/bin/env python3

# Copyright (2020) Katie Jones. All rights reserved.
"""Command-line script entrypoints."""

import sys

from .main import main


def main_script():
    """Main elevation to STL script."""
    return main(sys.argv)
