#!/usr/bin/env python3

# Copyright (2020) Katie Jones. All rights reserved.

from distutils.core import setup

setup(
    name='elevation_to_stl',
    version='alpha',
    description='Convert elevation data to STL file',
    author='Katie Jones',
    packages=['elevation_to_stl'],
    install_requires=[
        'SRTM.py',
        'pyyaml',
    ],
    entry_points={
        "console_scripts": [
            "elevation-to-stl=elevation_to_stl.command_line:main_script",
        ],
    },
)
