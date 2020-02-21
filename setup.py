# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:34:17 2020

"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="time-series-hda-ds-WS1920", # Replace with your own username
    version="0.0.9",
    author="project group",
    author_email="Karen.Schulz@stud.h-da.de",
    description="A small package including time series functionalities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kS-100/hda_project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=['__init__']
)
