import pygnuastro.fits
from numpy import ndarray
from os import remove

class TestFits:
  def test_simpleio(self):
    '''
    Performs simple input/output of a FITS image file using the
    img_read and img_write functions.
    '''
    image = pygnuastro.fits.img_read(filename = "test/psf.fits", hdu="1")
    # Confirm that the image data read is in a NumPy array
    assert isinstance(image, ndarray) == True
    # Passes if img_write returns True
    assert pygnuastro.fits.img_write(data = image, filename = "test/simpleio.fits") == True
    remove("test/simpleio.fits")