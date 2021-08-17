#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name = 'FEniCS_ii',
      version = '0.5',
      description = 'Interpreter approach to mixed-dimensional problems in FEniCS',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/fenics_ii.git',
      packages=find_packages(),
      include_package_data=True
)
