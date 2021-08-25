from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

setup (
	name = 'sensemaker_nyx', 
	version = '0.0.9',
	description = 'Nyx aka sensemaker',
	author = 'Andriy Kharchenko',
	url = 'https://github.com/friskluft/nyx', 
	package_dir={'':'src'},
	packages = find_packages(where='src'),
	python_requires = '>=3.6'
)
