***********
Basic Usage
***********

pyGnuastro is divided into separate modules(`Overview of the Python Modules <modules/overview.rst>`_)
and each module can be imported using the ``'.'`` after the package name as::
``import pygnuastro.MODULE_NAME``.

For Example

.. code-block:: python

  import pygnuastro.cosmology
  # Calling the age function in the cosmology module
  print(pygnuastro.cosomology.age(5))

Most functions also provide passing arguments as keywords.

.. code-block:: python
  
  import pygnuastro.cosmology as cosmo
  proper_distance = cosmo.proper_distance(H0 = 65, Z = 5,
                                          omatter = 0.3,
                                          olambda = 0.2,
                                          oradiation = 0.5)

For functions which are supposed to return or pass a NumPy array, we take
an example of a simple program which reads an input FITS image and stores
it as a NumPy array, then

.. code-block:: python

  import pygnuastro.fits

  # Here 'img' will be a NumPy array.
  img = pygnuastro.fits.img_read(filename = "input.fits", hdu = "0")

  # The img_write function returns True if the write was successful
  if pygnuastro.fits.img_write(img, "output.fits"):
    print("Write successful.")
  else:
    print("Error in writing file")
