#!/bin/bash

##########################################################################
# 1. Builds the wheel of the Python package.
# 2. Run Auditwheel to include the external dependecies in the wheel.
# 3. Use Pytest to run tests on the built wheels.
# 4. Cleans the artifacts generated during building.
##########################################################################



# Called inside the manylinux image
echo "Started $0 $@"
set -e -u -x
# Version of Gnuastro that is linked to the current version of pyGnuastro.
GAL_VERSION=0.18
source=/io
wheelsdir=wheelhouse




# Builds the .whl files for every CPython version using bdist_wheel.
function build_wheels(){
    cd $source/
    # When building wheels for the maneage docker image. Since Gnuastro
    # is statically built in maneage, the only files required are the
    # gnuastro library(libgnuastro.so) and its headers. Both of these are
    # stored in the ildir(install dir).
    ildir="/home/maneager/gnuastro-$GAL_VERSION"
    for PYBIN in /opt/python/cp*/bin; do
        "${PYBIN}/pip3" install -r $source/dev-requirements.txt
        "${PYBIN}/python3" $source/setup.py build_ext -I$ildir/include -lgnuastro -L$ildir/lib --rpath=$ildir/lib
        "${PYBIN}/python3" $source/setup.py bdist_wheel -d $wheelsdir/
    done
}





# Bundle external shared libraries into the wheels
function repair_wheels(){
    for whl in $source/$wheelsdir/*.whl; do
        # Shows external shared libraries that the wheel depends upon.
        if ! auditwheel show "$whl"; then
            echo "Skipping non-platform wheel $whl"
        else
            # Copies the external shared libraries into the wheel itself
            # and automatically modifies the appropriate RPATH entries
            # such that these libraries will be picked up at runtime.
            auditwheel repair "$whl" --plat "$PLAT" -w $source/$wheelsdir/
        fi
    done
}





# Install packages and test
function run_tests(){
    cd $source/
    for PYBIN in /opt/python/cp*/bin; do
        "${PYBIN}/pip3" install pygnuastro --no-index -f $source/$wheelsdir
        "${PYBIN}/python3" -m pytest
    done
}





function show_wheels(){
    ls -l $source/$wheelsdir/*.whl
}





function clean_system(){
    cd /io
    # Remove any libraries installed
    rm -rf build/ pygnuastro.egg-info/
    # Remove platform specific wheels
    rm -fv $wheelsdir/*-linux_*.whl
}





build_wheels
repair_wheels
run_tests
clean_system
show_wheels
