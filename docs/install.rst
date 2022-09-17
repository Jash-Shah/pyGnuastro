************
Installation
************

Installing ``pygnuastro``
=========================

Python packages can be distributed in mainly two ways, a `built distribution
<https://packaging.python.org/en/latest/glossary/#term-Built-Distribution>`_
and a `source distribution
<https://packaging.python.org/en/latest/glossary/#term-Source-Distribution-or-sdist>`_.
Built distributions are available in the form of
`wheels <https://realpython.com/python-wheels/>`_. Wheels offer a *faster
installation, smaller size and better maintainability*. By default pip will always
choose a wheel file over sdist file, unless you are using a very recent version of
Python, if a new version of ``pygnuastro`` has just been released, or if you are building
``pygnuastro`` for a platform that is not common. Note that in this case you will need a
C compiler (e.g., ``gcc`` or ``clang``) and the external dependencies to be installed
(see `Building from source`_ below) for the installation to succeed.

Using pip (PyPI)
----------------

To install ``pygnuastro`` with `pip <https://pypi.org/project/pip/>`_, run::

    pip install pygnuastro

If you want to make sure none of your existing dependencies get upgraded, you
can also do::

    pip install pygnuastro --no-deps


If you get a ``PermissionError`` this means that you do not have the required
administrative access to install new packages to your Python installation. In
this case you may consider using the ``--user`` option to install the package
into your home directory. You can read more about how to do this in the `pip
documentation <https://pip.pypa.io/en/stable/user_guide/#user-installs>`_.

Alternatively, if you wish to use ``pygnuastro`` in a project, or wish to
isolate the  ``pygnuastro`` package for a different purposes, consider installing
it in a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ instead.

Do **not** install ``pygnuastro`` using ``sudo``
unless you are fully aware of the risks.


Testing an Installed ``pygnuastro``
-----------------------------------
 
**TBA after adding a test module**. Currently `pytest <https://docs.pytest.org/en/7.1.x/getting-started.html#get-started>`_
can be used by running ``pytest`` in the top of the
`pyGnuastro source <https://github.com/Jash-Shah/pyGnuastro>`_ after having installed ``pyGnuastro``.


.. _source_dist_installation:

Building from Source
====================

.. warning::

  Source distribution installtions are not recommended since they require the user to
  have all the dependencies of pyGnuastro also installed.

The source distribution of ``pygnuastro`` is available in the form of a ``tar`` and
``zip`` files. The source distribution contains the projects source tree, which you can
find on `GitHub <https://github.com/Jash-Shah/pyGnuastro>`_. In order to install the package








Requirements
============

``pygnuastro`` has the following strict requirements:

- `Python`_ |minimum python version| or later

- `Numpy`_ |minimum numpy version| or later


.. _prereqs:

Prerequisites
-------------

You will need a compiler suite and the development headers for Python in order
to build ``pygnuastro``. Since pyGnuastro is a wrapper over the Gnuastro Library
the other essential requirement is `Gnuastro`_. It can be either built from source
or `installed using a supported package manager <https://repology.org/project/gnuastro/versions>`_
. To know more about how to build and install ``Gnuastro`` go through the `Installation chapter
<https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Installation.html>`_
in the `Gnuastro Book`_

Prerequisites for Linux
-----------------------

On Linux, using the package manager for your distribution will usually be the
easiest route to making sure you have the prerequisites to build ``pygnuastro``. In
order to build from source, you will need the Python development
package for your Linux distribution, as well as pip.

For Debian/Ubuntu::

    sudo apt-get install python3-dev python3-numpy-dev python3-setuptools

For Fedora/RHEL::

    sudo yum install python3-devel python3-numpy python3-setuptools


Prerequisites for Mac OS X
--------------------------

On MacOS X you will need the XCode command line tools which can be installed
using::

    xcode-select --install

Follow the onscreen instructions to install the command line tools required.
Note that you do **not** need to install the full XCode distribution (assuming
you are using MacOS X 10.9 or later).

The `instructions for building NumPy from source
<https://numpy.org/doc/stable/user/building.html>`_ are a good
resource for setting up your environment to build Python packages.

Obtaining the Source Packages
-----------------------------

Source Packages
^^^^^^^^^^^^^^^

The latest stable source package for ``pygnuastro`` can be `downloaded here
<https://pypi.org/project/pygnuastro>`_.

Development Repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of ``pygnuastro`` can be cloned from GitHub
using this command::

   git clone https://github.com/Jash-Shah/pyGnuastro.git


Building and Installing
-----------------------

To build and install ``pygnuastro`` (from the root of the source tree) assuming
you have all the :ref:`prereqs` installed::

    python3 setup.py build -I/path/to/gnuastro-installed-headers -L/path/to/gnuastro-installed-lib
    python3 setup.py install

.. note::

  Since ``pygnuastro`` depends on ``Gnuastro``, while building the python
  package, we need to provide the include path(``-I``) and the library path(``-L``) (See 
  `Directory Options <https://gcc.gnu.org/onlinedocs/gcc/Directory-Options.html#Directory-Options>`_)
  to link to the `Gnuastro Library`_ and its installed `headers 
  <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Headers.html>`_. The
  include path should be the path of the dirctory where the Gnuastro Library headers
  are installed(usually ``/usr/local/include``). The library path(usually
  ``/usr/local/lib``) is the path to the directory where the Gnuastro Library
  (``libgnuastro.so``) is located. See the `Installation Directory
  <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Installation-directory.html>`_
  chapter in the Gnuastro Book for more information on how and where to install
  Gnuastro.

  If Gnuatro was configured with no ``--prefix`` option or with it set to ``/usr/local/include``
  then no flags need to be passed to the  ``python3 setup.py build`` command.


Troubleshooting
---------------

If you get an error mentioning that you do not have the correct permissions to
install ``pygnuastro`` into the default ``site-packages`` directory, you can try
installing with::

    python3 setup.py install --user

which will install into a default directory in your home directory.

.. _for_developers:

For Developers
==============

The ``pygnuastro`` wheels ship with the external dependencies(`Gnuastro library`_).
This is done, so that users don't need to install and setup ``Gnuastro`` and can just
run ``pip install pygnuastro``.

However, this means deveopers who would like to tinker with the ``pygnuastro`` code are
tied with the version of ``Gnuastro`` in the wheel file. This can lead to scenarios
where a deveoper might want to, say add a python wrapper for a Gnuastro Library fucntion.
However, the version of the Gnuastro Library shipped in the wheel file does not have that
fucntion yet.

The best way to overcome this for developers, is to use the :ref:`source_dist_installation`.
