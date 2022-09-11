#ifndef PYTHON_H_
#define PYTHON_H_

#include <gnuastro/data.h>





/*************************************************************
 **************           Type codes           ***************
 *************************************************************/
int
gal_python_type_to_numpy(uint8_t type);

uint8_t
gal_python_type_from_numpy(int type);

#endif