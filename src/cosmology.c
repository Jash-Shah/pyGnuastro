#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <gnuastro/cosmology.h>



// Default values
// ==============
#define H0_DEFAULT 67.66
#define OLAMBDA_DEFAULT 0.6889
#define OMATTER_DEFAULT 0.3111
#define ORADIATION_DEFAULT 0.000

/* The names of the arguments as a static array.
  So that they can be accessed as keyword arguments in Python. */
static char *
kwlist[] = {"z", "H0", "olambda", "omatter", "oradiation", NULL};




















// Functions
// =========
static PyObject *
age(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
   i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_age(z, H0, o_lambda_0, o_matter_0,
                          o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
angular_distance(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_angular_distance(z, H0, o_lambda_0, o_matter_0,
                                       o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
comoving_volume(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
  /* The arguments passed don't correspond to the format
     described */
    return NULL;

  res = gal_cosmology_comoving_volume(z, H0, o_lambda_0, o_matter_0,
                                      o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
critical_density(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_critical_density(z, H0, o_lambda_0, o_matter_0,
                                       o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
distance_modulus(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_distance_modulus(z, H0, o_lambda_0, o_matter_0,
                                       o_radiation_0);

  return PyFloat_FromDouble(res);

}





static PyObject *
luminosity_distance(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_luminosity_distance(z, H0, o_lambda_0, o_matter_0,
                                          o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
proper_distance(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
     i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_proper_distance(z, H0, o_lambda_0, o_matter_0,
                                      o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
to_absolute_mag(PyObject *self, PyObject *args, PyObject *keywds)
{
  double z, res;
  double H0 = H0_DEFAULT;
  double o_lambda_0 = OLAMBDA_DEFAULT;
  double o_matter_0 = OMATTER_DEFAULT;
  double o_radiation_0 = ORADIATION_DEFAULT;

  /* "d|ddd" indicates that only the first argument
      i.e z is the required, and rest are optional args. */
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "d|dddd", kwlist,
                                   &z, &H0, &o_lambda_0, &o_matter_0,
                                   &o_radiation_0))
    /* The arguments passed don't correspond to the format
       described */
    return NULL;

  res = gal_cosmology_to_absolute_mag(z, H0, o_lambda_0, o_matter_0,
                                      o_radiation_0);

  return PyFloat_FromDouble(res);
}





static PyObject *
velocity_from_z(PyObject *self, PyObject *args)
{
  double z, vel;

  /* Not providing keyword arguments, since this function
     takes only one argument. */
  if (!PyArg_ParseTuple(args, "d", &z))
    return NULL;

  vel = gal_cosmology_velocity_from_z(z);

  return PyFloat_FromDouble(vel);
}





static PyObject *
z_from_velocity(PyObject *self, PyObject *args)
{
  double z, vel;

  /* Not providing keyword arguments, since this function
     takes only one argument. */
  if (!PyArg_ParseTuple(args, "d", &vel))
    return NULL;

  z = gal_cosmology_z_from_velocity(vel);

  return PyFloat_FromDouble(z);
}




















// Method Table
// ============
/* Define all the methods, with their
   name, function pointer, argument type and docstring. */
static PyMethodDef
CosmologyMethods[] =
{
  {
    "age",
    (PyCFunction)(void (*)(void))age,
    METH_VARARGS | METH_KEYWORDS,
    "Returns the age of the universe at redshift z in "
    "units of Giga years."
  },
  {
    "angular_distance",
    (PyCFunction)(void (*)(void))angular_distance,
    METH_VARARGS | METH_KEYWORDS,
    "Return the angular diameter distance to an "
    "object at redshift z in units of Mega parsecs."
  },
  {
    "comoving_volume",
    (PyCFunction)(void (*)(void))comoving_volume,
    METH_VARARGS | METH_KEYWORDS,
    "Returns the comoving volume over 4pi stradian "
    "to z in units of Mega parsecs cube."
  },
  {
    "critical_density",
    (PyCFunction)(void (*)(void))critical_density,
    METH_VARARGS | METH_KEYWORDS,
    "Returns the critical density at redshift z in "
    "units of g/cm3."
  },
  {
    "distance_modulus",
    (PyCFunction)(void (*)(void))distance_modulus,
    METH_VARARGS | METH_KEYWORDS,
    "Return the distance modulus at redshift z "
    "(with no units)."
  },
  {
    "luminosity_distance",
    (PyCFunction)(void (*)(void))luminosity_distance,
    METH_VARARGS | METH_KEYWORDS,
    "Return the luminosity diameter distance to an "
    "object at redshift z in units of Mega parsecs."
  },
  {
    "proper_distance",
    (PyCFunction)(void (*)(void))proper_distance,
    METH_VARARGS | METH_KEYWORDS,
    "Returns the proper distance to an object at "
    "redshift z in units of Mega parsecs."
  },
  {
    "to_absolute_mag",
    (PyCFunction)(void (*)(void))to_absolute_mag,
    METH_VARARGS | METH_KEYWORDS,
    "Return the conversion from apparent to absolute "
    "magnitude for an object at redshift z. This "
    "value has to be added to the apparent magnitude "
    "to give the absolute magnitude of an object at "
    "redshift z."
  },
  {
    "velocity_from_z",
    velocity_from_z,
    METH_VARARGS,
    "Return the velocity (in km/s) corresponding to "
    "the given redshift (z)."
  },
  {
    "z_from_velocity",
    z_from_velocity,
    METH_VARARGS,
    "Return the redshift corresponding to the given "
    "velocity (v in km/s)."
  },
  {NULL, NULL, 0, NULL} /* Sentinel */
};




















// Module Definition and Initialization
// ===================================
static struct PyModuleDef
cosmology =
{
  PyModuleDef_HEAD_INIT,
  "cosmology",
  "This library does the main cosmological calculations that "
  "are commonly necessary in extra-galactic astronomical "
  "studies. The main variable in this context is the redshift "
  "(z). The cosmological input parameters in the functions "
  "below are H0, o_lambda_0, o_matter_0, o_radiation_0 which "
  "respectively represent the current (at redshift 0) "
  "expansion rate (Hubble constant in units of km/sec/Mpc), "
  "cosmological constant (Λ), matter and radiation densities.",
  -1,
  CosmologyMethods
};



PyMODINIT_FUNC
PyInit_cosmology(void)
{
  PyObject *module;
  module = PyModule_Create(&cosmology);

  /* Error handling */
  if(module == NULL)
    return NULL;

  return module;
}