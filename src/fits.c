#define PY_SSIZE_T_CLEAN
// This has to be defined here to avoid NumPy deprecation messages.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>

#include <gnuastro/fits.h>
#include <gnuastro/python.h>

#include <numpy/arrayobject.h>




















// Functions
// =========
static PyObject *
img_read(PyObject *self, PyObject *args, PyObject *keywds)
{
  char *filename, *hdu;
  PyObject *out = NULL;
  gal_data_t *image = NULL;
  // Default values of minmapsize and quietmap
  int minmapsize = -1, quietmap = 1;

  /* The names of the arguments as a static array.
  So that they can be accessed as keyword arguments in Python. */
  static char*
  kwlist[] = {"filename", "hdu", "minmapsize",
               "quietmap", NULL};

  /* "ss|ii" indicates that only the first two arguments
      i.e filename and hdu are required, rest are optional. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|ii", kwlist,
                                   &filename, &hdu,
                                   &minmapsize, &quietmap))
    /* The arguments passed don't correspond to the format
       described */                                   
    return NULL;

  /* Reading the image */
  image = gal_fits_img_read(filename, hdu, minmapsize, quietmap);

  /* Since dims needs to be a pointer to a const. */
  npy_intp* const dims = (npy_intp *)image->dsize;

  out = PyArray_SimpleNewFromData(image->ndim, dims,
                                  gal_python_type_to_numpy(image->type),
                                  (float *)image->array);
  

  return out;
}





static PyObject *
img_write(PyObject *self, PyObject *args, PyObject *keywds)
{
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
  // printf("Arguments parsed\n");

  /* Converting the received Python array data to NumPy array. */
  data_arr = (PyArrayObject *)PyArray_FROM_OT(img_data, NPY_FLOAT32);
  // printf("Numpy Data Array initialized\n");

  /* Using the metadata obtained from the NumPy array to allocate a gal_data_t 
     structure which will be used for writing to the output image. */
  data = gal_data_alloc(PyArray_DATA(data_arr), gal_python_type_from_numpy(PyArray_TYPE(data_arr)),
                        PyArray_NDIM(data_arr), (size_t *)PyArray_DIMS(data_arr),
                        NULL, 0, -1, 1, NULL, NULL, NULL);
  // printf("gal_data_alloc succedded\n");

  gal_fits_img_write(data, filename, headers, program_string);

  // printf("%s created!\n",filename);

  gal_data_free(data);

  /* Returns acknowledgement if write was successful. */
  return Py_True;

}





static PyMethodDef
FitsMethods[] = {
                  {
                    "img_read",
                    (PyCFunction)(void (*)(void))img_read,
                    METH_VARARGS | METH_KEYWORDS,
                    "Reads an image."
                  },
                  {
                    "img_write",
                    (PyCFunction)(void (*)(void))img_write,
                    METH_VARARGS | METH_KEYWORDS,
                    "Writes an image."
                  },
    {NULL, NULL, 0, NULL}, /* Sentinel */
};





static struct PyModuleDef
fits = {
          PyModuleDef_HEAD_INIT,
          "fits",
          "FITS Module",
          -1,
          FitsMethods
       };



PyMODINIT_FUNC
PyInit_fits(void)
{
  PyObject *module;
  module = PyModule_Create(&fits);
  if(module==NULL) return NULL;

  import_array();

  if(PyErr_Occurred()) return NULL;

  return module;
}