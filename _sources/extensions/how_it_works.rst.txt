*************************
How It Works - An Example
*************************

While the `Python wrappers <wrappers>` explains the formation of the
modules and the pyGnuastro package, to get a more clear idea of how
the interfacing between the `Gnuastro Library`_ and pyGnuastro we take
the example of one of the most frequent use cases of an astronomy library
i.e. **Reading and Writing with a FITS image**.

A basic example of an I/O operation is to read data from a FITS(``.fits``)
image and writing the same data to an output FITS image. We can perform
this task using ``pygnuastro`` as follows:

.. code-block:: python

  import pygnuastro.Fits

  ###################### READING ######################
  # Here 'in' will be a NumPy array.
  in = pygnuastro.fits.img_read(filename = "input.fits",
                                hdu = 0)

  ###################### WRITING ######################
  if gnuastro.fits.img_write(in, "output.fits"):
    print("Write successful.")
  else:
    print("Error in writing file")

The flow of the operations here and their interaction with ``Gnuastro``
can be described as follows:

1. Call the ``pygnuastro.fits.img_read()`` function from Python, passing the ``filename`` and
   ``hdu`` as arguments.
2. The C wrapper function parses these arguments and stores them in C variables of
   appropriate data type (here ``filename`` and ``hdu`` are strings).`
3. Call the `gal_fits_img_read() <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/FITS-arrays.html#index-gal_005ffits_005fimg_005fread>`_
   with these parsed arguments. This function returns a `gal_data_t <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Generic-data-container.html>`_
   type.
4. Convert the returned value to a `PyArrayObject <https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.PyArrayObject>`_
   using one of the `NumPy array creation functions <https://numpy.org/doc/stable/reference/c-api/array.html#creating-arrays>`_
   and return it.
5. Send the received ``numpy.array`` object as an argument to ``pygnuastro.fits.img_write()``
   alongwith the output FITS image ``filename``.
6. The C wrapper function for the write method parses these arguments like in the case
   of reading, this time storing the NumPy array as a ``PyArrayObject``.
7. Use `gal_data_alloc() <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Dataset-allocation.html#index-gal_005fdata_005falloc>`_
   to convert the input data-array from a ``PyArrayObject`` to ``gal_data_t``.
8. Use `gal_fits_img_write() <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/FITS-arrays.html#index-gal_005ffits_005fimg_005fwrite>`_
   to then write this data to the output FITS ``filename`` and return the status of the write.
