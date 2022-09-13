#define PY_SSIZE_T_CLEAN
/* This macro needs to be defined before including any NumPy headers
   to avoid the compiler from raising a warning message. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* This has to be included so that the NumPy C-API can be included
   in multiple files for a single-extension module. The import_array()
   function included in the module initialization, is declare static
   if this macro is not defined. This means that the NumPy C-API doesn't
   extend to other files (like utils.c) without this macro. */
#define PY_ARRAY_UNIQUE_SYMBOL pygnuastro_ARRAY_API
#include <numpy/arrayobject.h>

#include <Python.h>

#include <gnuastro/fits.h>

/* Since the Python interface(python.h) was added to Gnuastro in version
   0.19(unreleased as of yet), we include the interface functions from
   the 'utils' directory instead. */
#include "utils/utils.h"



















// Functions
// =========
static PyObject *
img_read(PyObject *self, PyObject *args, PyObject *keywds)
{
  int npy_type;
  char *filename, *hdu;
  PyObject *out = NULL;
  gal_data_t *image = NULL;
  /* Default values of minmapsize and quietmap */
  int minmapsize = -1, quietmap = 1;

  /* The names of the arguments as a static array.
     So that they can be accessed as keyword arguments in Python. */
  static char*
  kwlist[] = {"filename", "hdu", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss", kwlist,
                                   &filename, &hdu))
    /* The arguments passed don't correspond to the format
       described */                                   
    return NULL;

  /* Reading the image */
  image = gal_fits_img_read(filename, hdu, minmapsize, quietmap);

  /* Since dims needs to be a pointer to a const. */
  npy_intp* const dims = (npy_intp *)image->dsize;
  npy_type = gal_python_type_to_numpy(image->type);
  if (npy_type == 0)
    {
      Py_XDECREF(out);
      PyErr_SetString(PyExc_TypeError,
                      "Type code of the data read from file is "
                      "not convertible to a NumPy type.");
      return NULL;                      
    }

  out = PyArray_SimpleNewFromData(image->ndim, dims,
                                  npy_type,
                                  (float *)image->array);
  
  return out;
}





static PyObject *
img_write(PyObject *self, PyObject *args, PyObject *keywds)
{
  int gal_type;
  gal_data_t *data = NULL;
  PyObject *img_data = NULL;
  PyArrayObject *data_arr = NULL;
  char *filename, *program_string;
  /* Default value for program_string */
  program_string = "FITS Program";

  static char *
  kwlist[] = {"data", "filename", "program_string"};

  if(!PyArg_ParseTupleAndKeywords(args, keywds, "Os|s", kwlist,
                                  &img_data, &filename, &program_string))
    return NULL;

  /* Converting the received Python array data to NumPy array. */
  data_arr = (PyArrayObject *)PyArray_FROM_OT(img_data, NPY_FLOAT32);
  gal_type = gal_python_type_from_numpy(PyArray_TYPE(data_arr));
  if (gal_type == 0)
   {
      Py_XDECREF(img_data);
      Py_XDECREF(data_arr);
      PyErr_SetString(PyExc_TypeError,
                      "Type code of the data to be written to file"
                      "is not convertible to a Gnuastro type.");
      return NULL;                      
   }

  /* Using metadata obtained from the NumPy array to allocate a gal_data_t
     structure which will be used for writing to the output image. */
  data = gal_data_alloc(PyArray_DATA(data_arr),
                        gal_type,
                        PyArray_NDIM(data_arr),
                        (size_t *)PyArray_DIMS(data_arr),
                        NULL, 0, -1, 1, NULL, NULL, NULL);

  gal_fits_img_write(data, filename, NULL, program_string);

  gal_data_free(data);

  /* Returns acknowledgement if write was successful. */
  return Py_True;

}




















// Method Table
// ============
/* Define all the methods, with their
   name, function pointer, argument type and docstring. */
static PyMethodDef
FitsMethods[] =
{
  {
    "img_read",
    (PyCFunction)(void (*)(void))img_read,
    METH_VARARGS | METH_KEYWORDS,
    "Reads the contents of the 'hdu' extension/HDU of "
    "'filename' into a NumPy array and returns it. Note "
    "that this function only reads the main data within "
    "the requested FITS extension, the WCS will not be "
    "read into the returned dataset."
  },
  {
    "img_write",
    (PyCFunction)(void (*)(void))img_write,
    METH_VARARGS | METH_KEYWORDS,
    "Write the input 'data' into the FITS file named "
    "'filename'. Also, add the program's name('program "
    "string') to the newly created HDU/extension."
  },
  {NULL, NULL, 0, NULL}, /* Sentinel */
};




















// Module Definition and Initialization
// ===================================
static struct PyModuleDef
fits =
{
  PyModuleDef_HEAD_INIT,
  "fits",
  "The FITS format is the most common format to store data (images and "
  "tables) in astronomy.  The CFITSIO library already provides a very "
  "good low-level collection of functions for manipulating FITS data. "
  "pyGnuastro's FITS module provides access to the Gnuastro FITS library "
  "functions which provide wrappers for CFITSIO functions. With these, "
  "it is much easier to read, write, or modify FITS file data, header "
  "keywords and extensions.",
  -1,
  FitsMethods
};



PyMODINIT_FUNC
PyInit_fits(void)
{
  PyObject *module;
  module = PyModule_Create(&fits);

  /* Error handling */
  if(module==NULL)
    return NULL;

  if (PyArray_API == NULL)
    import_array();
  
  if(PyErr_Occurred()) return NULL;

  return module;
}
