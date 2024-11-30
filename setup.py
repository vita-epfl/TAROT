#!/usr/bin/env python
from setuptools import setup

setup(
    name="tarot",
    version="0.0.1",
    description="TAROT: Targeted Data Selection via Optimal Transport Distance Minimization",
    author="Lan Feng",
    author_email="lan.feng@epfl.ch",
    license_files=("LICENSE.txt",),
    packages=["tarot"],
    install_requires=[
    ],
    include_package_data=True,
)
