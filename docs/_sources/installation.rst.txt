.. toctree::
    :caption: Contents

Installation
============

Requirements
------------
SpecUFEx depends on numpy, scipy, and h5py.
It has been tested in python 3.7 (so far).

Installing in a virtual environment
-----------------------------------
We recommend installing this package in a virtual environment.
We typically use conda but virtualenv should work as well.
While this package does not have a conda-specific
installer yet, you can install it into a conda environment using pip.

Install using pip
-----------------
Once you have activated your environment, clone this repository
to your computer, cd to the directory, and use `pip` to install the package.::

    git clone https://github.com/nategroebner/specufex.git
    cd specufex
    pip install -e .