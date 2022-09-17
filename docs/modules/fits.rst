**************************
Fits (``pygnuastro.fits``)
**************************

The `FITS format  <https://en.wikipedia.org/wiki/FITS>`_ is the most
common format to store data (images and tables) in astronomy. 
The `CFITSIO library <https://heasarc.gsfc.nasa.gov/fitsio/>`_
already provides a very good low-level collection of functions
for manipulating FITS data. pyGnuastro's FITS module provides
access to the `Gnuastro FITS library
<https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/FITS-files.html>`_
functions which provide wrappers for CFITSIO functions. With these, 
it is much easier to read, write, or modify FITS file data, header 
keywords and extensions.

.. note::

  This module is currently under development and more functions will be added
  in the future.

.. autofunction:: pygnuastro.fits.img_read
.. autofunction:: pygnuastro.fits.img_write