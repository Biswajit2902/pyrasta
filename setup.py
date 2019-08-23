#!/usr/bin/env python
__author__ = 'Biswajit Satapathy'

import os, sys, shutil, compileall
from subprocess import call
from setuptools import setup, find_packages
from shutil import copytree, rmtree, ignore_patterns, copy, copyfile

import rasta

__this__='pyrasta'
__pypath__ = sys.executable
__pybin__="/".join(__pypath__.split('/')[0:-1])
__envpath__="/".join(__pypath__.split('/')[0:-2])
__rootdir__=os.getcwd()


srcs=[]
scripts=[]
scripts_cron=[]
scripts_stop=[]

if (sys.version_info[0] == 2 and sys.version_info[1] < 7) or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
    sys.exit('Sorry, Python < 2.7 is not supported')

install_requires=[]
with open(os.path.join(__rootdir__,'requirement.txt'),'r') as fp:
    install_requires = [d.strip() for d in fp.readlines()]
    fp.close()

setup(
    name = "pyrasta",
    version = rasta.version.version,
    author = "Biswajit Satapathy",
    author_email = "biswajit2902@gmail.com",
    description = ("PLP and RASTA MFCC in Python"),
    license = ("Ozonetel"),
    keywords = "RASTA filtering, RASTA PLP, MFCC, Audio Processing, Feature Engineering",
    url = "",
    dependency_links=[],
    packages = ["rasta"],
    package_data={
        'rasta': ['*.pyc']
    },
    scripts = scripts,
    install_requires = install_requires
)
