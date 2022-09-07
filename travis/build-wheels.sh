#!/bin/bash

##########################################################################
# 1. Sets up Gnuastro and its dependencies on the system.
# 2. Build the wheel of the Python package.
# 3. Run Auditwheel to include the external dependecies in the wheel.
# 4. Use Pytest to run tests on the built wheels.
# 5. Cleans the artifacts generated during building.
##########################################################################



# Called inside the manylinux image
echo "Started $0 $@"

set -e -u -x
# The version of Gnuastro that is linked to the current version of PyGnuastro.
GAL_VERSION=latest




# Downloads, builds and installs gnuastro.
function setup_gnuastro(){
    cd /io
    # Make the version of Python same as version of Gnuastro.
    wget -c http://ftpmirror.gnu.org/gnuastro/gnuastro-$GAL_VERSION.tar.gz \
    -O - | tar -xz 
    cd gnuastro*
    # Not added --disable-shared since its giving errors in 
    # CentOS. Check out bug#62904.
    ./configure
    make -j$(nproc) -s
    make check -j$(nproc)
    make install
}





# Downloads, builds and installs cfitsio.
function setup_cfitsio(){
    cd /io
    wget -c http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz \
    -O - | tar -xz
    cd cfitsio*
    # Build instructions taken from:
    # https://www.gnu.org/software/gnuastro/manual/html_node/CFITSIO.html
    ./configure --prefix=/usr/local --enable-sse2 --enable-reentrant
    make -j$(nproc) -s
    # Skipping these steps, since fpack and funpack
    # are not requirements for building Gnuastro library.
    # make utils
    # export LD_LIBRARY_PATH="$(pwd):$LD_LIBRARY_PATH"
    # ./testprog > testprog.lis
    # diff testprog.lis testprog.out
    # cmp testprog.fit testprog.std
    make install
}





# Downloads, builds and installs wcslib.
function setup_wcslib(){
    cd /io
    wget -c http://www.atnf.csiro.au/people/mcalabre/WCS/wcslib-7.11.tar.bz2 \
    -O - | tar xj wcs*
    cd wcslib*
    # Build instructions taken from:
    # https://www.gnu.org/software/gnuastro/manual/html_node/WCSLIB.html
    ./configure LIBS="-pthread -lm" --without-pgplot     \
              --disable-fortran --enable-shared
    make -j$(nproc) -s
    make check -j$(nproc)
    make install
}





# This is to setup Gnuastro on a Debian based system.
# function prepare_system_debian(){
#     apt-get update -y
#     # Install the system packages required by our library
#     # topcat is not installed since its not available in Debian 9.
#     apt-get install ghostscript libtool-bin libjpeg-dev wget   \
#                     libtiff5-dev libgit2-dev curl lzip wget saods9 \
#                     libgsl-dev libcfitsio-dev wcslib-dev -y
#     echo "Python versions found: $(cd /opt/python && echo cp* \
#           | sed -e 's|[^ ]*-||g')"
#     setup_gnuastro
# }





# For CentOS based systems.
function prepare_system_centos(){
    # Install as many dependencies as possible using the
    # package manager. Taken from 
    # https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Dependencies-from-package-managers.html#index-RHEL
    # ds9 and topcat are not available in CentOS7. Also they are
    # not required for building the library, so can be skipped.
    yum install ghostscript libtool libjpeg-devel     \
                libtiff-devel libgit2-devel lzip curl \
                gsl-devel wget -y
    # Update all packages installed.
    yum update -y
    # "/usr/local/lib" is not added to LD_LIBRARY_PATH by
    # default, so adding it here and performing ldconfig to
    # avoid any linking errors later on when building gnuastro.
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
    ldconfig

    setup_wcslib
    # Only for manylinux2014_x86_64, since
    # libcfitsio-dev is not avaialble in CentOS through yum.
    setup_cfitsio
    echo "Python versions found: $(cd /opt/python && echo cp* \
          | sed -e 's|[^ ]*-||g')"
    setup_gnuastro
}





# Builds the .whl files for every CPython version using bdist_wheel.
function build_wheels(){
    cd /io/
    # When building wheels for the maneage docker image
    ildir="$HOME/build/software/installed"
    for PYBIN in /opt/python/cp*/bin;   do
        "${PYBIN}/pip3" install -r /io/dev-requirements.txt
        "${PYBIN}/python3" /io/setup.py build_ext -I$ildir/include -L$ildir/lib --rpath=$ildir/lib
        "${PYBIN}/python3" /io/setup.py bdist_wheel -d wheelhouse/
    done
}





# Bundle external shared libraries into the wheels
function repair_wheels(){
    for whl in /io/wheelhouse/*.whl; do
        # Shows external shared libraries that the wheel depends upon.
        if ! auditwheel show "$whl"; then
            echo "Skipping non-platform wheel $whl"
        else
            # Copies the external shared libraries into the wheel itself
            # and automatically modifies the appropriate RPATH entries
            # such that these libraries will be picked up at runtime.
            auditwheel repair "$whl" --plat "$PLAT" -w /io/wheelhouse/
        fi
    done
}





# Install packages and test
function run_tests(){
    cd /io/
    for PYBIN in /opt/python/cp*/bin; do
        "${PYBIN}/pip3" install pygnuastro --no-index -f /io/wheelhouse || exit 1
        "${PYBIN}/python3" -m pytest
    done
}





function show_wheels(){
    ls -l /io/wheelhouse/*.whl
}





function clean_system(){
    cd /io
    # Remove any libraries installed
    rm -rf build/ pygnuastro.egg-info/
    # Remove platform specific wheels
    rm -fv wheelhouse/*-linux_*.whl
}





# prepare_system_centos
build_wheels
repair_wheels
run_tests
clean_system
show_wheels
