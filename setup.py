from setuptools import setup, find_packages

from codecs import open
from os import path

setup(
    name='pythia',
    version='1.0',
    description='This package is a tool to perform linear regression',
    url='https://github.com/UBC-MDS/Pythia',
    author='MDS students: Maud Boucherit, Teddy Haley, Cem Sinan Ozturk',
    keywords='linear regression pythia',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    license=open('LICENSE.md').read(),
    long_description=open('README.md').read(),
)
