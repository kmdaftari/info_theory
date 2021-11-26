# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

# this function writes the readme file as a string which is called by the long_description 
# in setup below

with open('README.md') as f:
    readme = f.read()


# this function is called by pip install to install the package named "info_theory"

setup(

    # name of package for import statement
    name = 'info_theory',

    # version of package, manually update every time package changes
    version = '0.0.0',

    # short description
    description = 'estimate information theoretic measures from empirical data',

    # long description
    long_description = readme,

    # do not include tests and documentation when importing package
    packages = find_packages(exclude = ('tests', 'docs', 'Jupyter_notebooks'))
)
