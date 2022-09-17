**********************************************
Python C Extension Modules and Python Wrappers
**********************************************

pyGnuastro is a low-level wrapper infrastructure over the `Gnuastro Library`_.
The source of the Python modules are ``C`` wrapper functions which are in
the ``src/`` directory. These functions are basicly `Python Extension Modules
<https://docs.python.org/3/extending/extending.html>`_, which internally make
use of the Gnuastro Library functions. The basic format of a wrapper function
is as follows:

.. code-block:: c

  #include <Python.h>

  static PyObject
  *function_name(PyObject *self, PyObject *args)
  {
    /* Initialize the C variables here which will be
       assigned the value from parsing the Python input arguments.
       This step parses the input args based on a format
       string and stores them in the C variables initialized above. */
    if(!PyArg_ParseTuple(args,"format_string", &var1, &var2, ...))
      return NULL;
    
    /* Now, we call the Gnuastro library function with those arguments */
    result = gal_function_name(var1, var2, ...);

    /* Before returning the result, we need to convert it to
       a PyObject again, so that it can be interprested by Python. */
    return PyFloat_FromDouble(result);
  }



  static PyMethodDef
  ModuleMethods[] =
  {
    /* When using only METH_VARARGS, the function should expect the
    Python-level parameters to be passed in as a tuple acceptable for
    parsing via PyArg_ParseTuple(). We can also add functionality to
    pass keyword arguments. */
    {"function_name", function_name, METH_VARARGS, "Perform a function."},
    .
    .
    .
    {NULL, NULL, 0, NULL} /* Sentinel */
  };

  static struct PyModuleDef
  Module =
  {
    PyModuleDef_HEAD_INIT,
    "MODULE_NAME",       /* name of module                   */
    "Module Definition", /* module documnetation, maybe NULL */
    -1,                  /* size of per-interpreter state of
                            the module, or -1 if the module
                            keeps state in global variables. */
    ModuleMethods
  };



  /*
  The initialization function must be named PyInit_name(), where name is
  the name of the module, and should be the only non-static item defined
  in the module file. PyMODINIT_FUNC declares the function as
  PyObject * return type and declares any special linkage declarations
  required by the platform.
  */
  PyMODINIT_FUNC
  PyInit_spam(void)
  {
    /* PyModule_Create(), which returns a module object, and inserts
    built-in function objects into the newly created module based upon
    the method table (an array of PyMethodDef structures) found in the
    module definition. */
    return PyModule_Create(&pygnuastromodule);
  }

For modules which deal with ``NumPy``, we make use of various functions
provided by the `NumPy C-API <https://numpy.org/doc/stable/reference/c-api/index.html>`_.
The core data structure of Gnuastro is the `gal_data_t <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Generic-data-container.html>`_.
If any function in the Gnuastro Library is passed or returns a value of
this type, then we make use of functions provided by the NumPy C-API, to convert it
to a ``NumPy`` array.