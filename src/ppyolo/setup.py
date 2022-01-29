#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ppyolo'],
    package_dir={'': 'scripts'},
    scripts=['scripts/pp_infer_remake.py']
)

setup(**d)
