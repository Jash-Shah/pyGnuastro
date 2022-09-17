import os
import platform
from numpy import get_include
from setuptools import setup, Extension





# Initialization
# ==============
# Get the absolute path for the current directory
here = os.path.abspath(os.path.dirname(__file__))



# Description of the Gnuastro Python package.
with open(here+"/README.rst") as readme:
  long_descp = readme.read()







# Path to the source files for the extension
# modules where the wrapper functions and NumPy
# converters are written.
src_dir = "src"

# For Debugging:
# include_dir = "/usr/include"
# lib_dir = "/usr/lib/gnuastro"





# These arguments will be common while initializing
# all Extension modules. Hence, can be defined here only once.
default_ext_args = dict(include_dirs=["/usr/local/include",
                                      get_include(), "."],
                        libraries=["gnuastro"],
                        library_dirs=["/usr/local/lib"])





# Extension Modules
# =================
# Each module is defined using its name, source
# file and the default arguments dictionary defined above.
cosmology = Extension(name='cosmology',
                      sources=[f'{src_dir}/cosmology.c'],
                      **default_ext_args)



fits = Extension(name='fits',
                 sources=[f'{src_dir}/fits.c',
                          f'{src_dir}/utils/utils.c'],
                 **default_ext_args)





# Setup
# =====
# This is the main funciton which builds the module.
# It uses the metadata passed as arguments to describe the Python Package
setup(
      # Name of the package as it appears on PyPI
      name="pygnuastro",
      version=f'0.0.1-dev2',
      #
      # Longer description of your project that represents
      # the body of text which users will see when they visit PyPI.
      long_description=long_descp,
      long_description_content_type="text/x-rst",
      #
      # Author Details
      author="Jash Shah",
      author_email="jash28582@gmail.com",
      #
      # Project Home-Page url
      url="https://github.com/Jash-Shah/pyGnuastro/",
      project_urls={
        "Manual": "https://jash-shah.github.io/pyGnuastro/index.html",
        "Issues": "https://github.com/Jash-Shah/pyGnuastro/issues",},
      #
      # Classifiers to help users find the project by categorizing it.
      classifiers=[
        # Maturity of Project(1 to 6)
        "Development Status :: 1 - Planning",
        #
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        # License Type
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        # Specifing the Python versions supported. These classifiers are
        # *not* checked by 'pip install'. See instead 'python_requires'.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"],
      #
      # This field adds keywords for your project which will appear on the
      # project page.
      keywords="astronomy, astrophysics, cosmology, space, science, \
                units, table, wcs, coordinate, fits, fitting",
      #
      # 'pip install' will check this and refuse to
      # install the project if the version does not match.
      python_requires=">=3.6, <4",
      #
      # Other packages that the project depends on to run
      install_requires=["numpy"],
      #
      # Additional groups of dependencies, which are not installed
      # by default, but users can install by giving appropriate arguments.
      #
      setup_requires=["numpy"],
      #
      extras_require={
        "dev": ["tox", "twine", "auditwheel", "pytest"],
        "test": ["tox", "pytest"],
      },
      #
      # This will be used as the base package name.
      ext_package="pygnuastro",
      ext_modules=[cosmology, fits])
