from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

setup (
	name = 'nyx', 
	version = '0.0.6',
	description = 'Faeton > Nyx > sensemaker',
	
	#py_modules = 'interface', 
	#package_dir = {'':'src', 'nyx':'./src/nyx'},
    package_dir={'':'src'},
	packages=find_packages(where='src'),
	
	python_requires='>=3.6'
)

