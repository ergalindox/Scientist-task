import setuptools
from pathlib import Path

this_directory = Path(__file__).parent

setuptools.setup(
    name = 'buynomics',
    version = '1.0.0',
    author = 'Gabriel Galindo',
    description = 'Generate model prediction',
    packages = setuptools.find_packages(), #find all packages which can be found by __init__.py
    install_requires = []
)