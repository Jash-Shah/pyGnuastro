/*********************************************************************
python -- Functions to assist Python wrappers using Gnuastro's library.
**********************************************************************/
#include <gnuastro/config.h>

/* This macro needs to be defined before including any NumPy headers
   to avoid the compiler from raising a warning message. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* This has to be included so that the NumPy C-API can be included
   here without the import_array() function.*/
#define PY_ARRAY_UNIQUE_SYMBOL pygnuastro_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

/* Python Interface headers. */
#include "utils.h"





/*************************************************************
 **************           Type codes           ***************
 *************************************************************/

/* Convert Gnuastro type to NumPy datatype. Currently only converting types
   directly compatible between the two. */
int
gal_python_type_to_numpy(uint8_t type)
{
  switch (type)
    {
    case GAL_TYPE_INT8:      return NPY_INT8;
    case GAL_TYPE_INT16:     return NPY_INT16;
    case GAL_TYPE_INT32:     return NPY_INT32;
    case GAL_TYPE_INT64:     return NPY_LONG;
    case GAL_TYPE_UINT8:     return NPY_UINT8;
    case GAL_TYPE_UINT16:    return NPY_UINT16;
    case GAL_TYPE_UINT32:    return NPY_UINT32;
    case GAL_TYPE_UINT64:    return NPY_UINT64;
    case GAL_TYPE_FLOAT32:   return NPY_FLOAT32;
    case GAL_TYPE_FLOAT64:   return NPY_FLOAT64;
    case GAL_TYPE_STRING:    return NPY_STRING;
    default:
      return GAL_TYPE_INVALID;
    }

  return GAL_TYPE_INVALID;
}





/* Convert Numpy datatype to Gnuastro type. Currently only converting types
   directly compatible between the two. */
uint8_t
gal_python_type_from_numpy(int type)
{
  switch (type)
    {
    case NPY_INT8:           return GAL_TYPE_INT8;
    case NPY_INT16:          return GAL_TYPE_INT16;
    case NPY_INT32:          return GAL_TYPE_INT32;
    case NPY_LONG:           return GAL_TYPE_INT64;
    case NPY_UINT8:          return GAL_TYPE_UINT8;
    case NPY_UINT16:         return GAL_TYPE_UINT16;
    case NPY_UINT32:         return GAL_TYPE_UINT32;
    case NPY_UINT64:         return GAL_TYPE_UINT64;
    case NPY_FLOAT32:        return GAL_TYPE_FLOAT32;
    case NPY_FLOAT64:        return GAL_TYPE_FLOAT64;
    case NPY_COMPLEX64:      return GAL_TYPE_COMPLEX64;
    case NPY_STRING:         return GAL_TYPE_STRING;
    default:
        return GAL_TYPE_INVALID;
    }
  return GAL_TYPE_INVALID;
}
