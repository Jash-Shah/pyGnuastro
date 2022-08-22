import os
from numpy import get_include
from setuptools import setup, Extension





# Initialization
# ==============
# Get the absolute path for the current directory
here = os.path.abspath(os.path.dirname(__file__))



# Description of the Gnuastro Python package.
with open(here+"/README.md") as readme:
  long_descp = readme.read()



# Get the paths to where the gnuastro library(libgnuastro),
# the source files(.h) and the extension modules(.c)
# are from environment variables defined in the Makefile.





# Path to the source files for the extension
# modules where the wrapper functions and NumPy
# converters are written. These will be in the source tree.
src_dir = "src"

# For Debugging:
# include_dir = "/usr/include"
# lib_dir = "/usr/lib/gnuastro"





# These arguments will be common while initializing
# all Extension modules. Hence, can be defined here only once.
default_ext_args = dict(include_dirs=["/usr/include",
                                      get_include()],
                        libraries=["gnuastro"],
                        library_dirs=["/usr/local/lib"])





# Extension Modules
# =================
# Each module is defined using its name, source
# file and the default arguments dictionary defined above.
cosmology = Extension(name='cosmology',
                      sources=[f'{src_dir}/cosmology.c'],
                      **default_ext_args)



# fits = Extension(name='fits',
#                  sources=[f'{src_dir}/fits.c'],
#                  **default_ext_args)





# Setup
# =====
# This is the main funciton which builds the module.
# It uses the metadata passed as arguments to describe the Python Package
setup(name="pygnuastro",
      version=f'0.18',
      long_description=long_descp,
      long_description_content_type="text/markdown",
      author="Mohammad Akhlaghi",
      author_email="mohammad@akhlaghi.org",
      url="http://www.gnu.org/software/gnuastro/manual/",
      project_urls={
        "Manual": "http://www.gnu.org/software/gnuastro/manual/",
        "Bug Tracker": "https://savannah.gnu.org/bugs/?group=gnuastro",},
      ext_package="pygnuastro", # This will be used as base package name.
      ext_modules=[cosmology])