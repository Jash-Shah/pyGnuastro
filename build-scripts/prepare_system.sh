#################################################################
# Prepares the system by setting up Gnuastro and its dependencies
#################################################################



# Version of Gnuastro that is linked to the current version of pyGnuastro.
GAL_VERSION=0.18
os_id=$(awk -F= '/^ID/{print $2}' /etc/os-release)
os_id_like=$(awk -F= '/^ID_LIKE/{print $2}' /etc/os-release)
builddir=/home/





# Downloads, builds and installs gnuastro.
function setup_gnuastro(){
    cd $builddir
    # Make sure the correct version of Gnaustro corresponding to this
    # version of pyGnuastro is installed.
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
    cd $builddir
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
    cd $builddir
    wget -c http://www.atnf.csiro.au/people/mcalabre/WCS/wcslib-7.11.tar.bz2 \
    -O - | tar xj wcs*
    cd wcslib*

    # Build instructions taken from:
    # https://www.gnu.org/software/gnuastro/manual/html_node/WCSLIB.html
    ./configure LIBS="-pthread -lm" --without-pgplot \
                --disable-fortran --enable-shared
    make -j$(nproc) -s
    make check -j$(nproc)
    make install
}





# Installs the dependencies of Gnuastro using package managers if possible.
# Currently only supports CentOS and Debian based systems.
function prepare_system(){
    # Install as many dependencies as possible using the
    # package manager. Taken from
    # https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Dependencies-from-package-managers.html#index-RHEL
    # ds9 and topcat are skipped since they arent required for
    # building the library.
    if [ "$os_id" = "centos" ]; then
      yum install ghostscript libtool libjpeg-devel     \
                  libtiff-devel libgit2-devel lzip curl \
                  gsl-devel wget -y
      # Update all packages installed.
      yum update -y
      # Only for manylinux2014_x86_64, since
      # libcfitsio-dev and wcslib-dev were not avaialble in CentOS
      #  through yum.
      setup_wcslib
      setup_cfitsio
    elif [ "$os_id_like" = "debian" ];
      # Update all packages installed.
      apt-get update -y
      # topcat is not installed since its not available in Debian 9.
      apt-get install ghostscript libtool-bin libjpeg-dev    \
                      libtiff5-dev libgit2-dev curl lzip wget \
                      libgsl-dev libcfitsio-dev wcslib-dev -y
    fi

    # "/usr/local/lib" is not added to LD_LIBRARY_PATH by
    # default, so adding it here and performing ldconfig to
    # avoid any linking errors later on when building gnuastro.
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"
    ldconfig

    echo "Python versions found: $(cd /opt/python && echo cp* \
          | sed -e 's|[^ ]*-||g')"
    setup_gnuastro
}



prepare_system
