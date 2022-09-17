****************
About pyGnuastro
****************

pyGnuastro started as a `task on Gnuastro's Savannah page <https://savannah.gnu.org/task/?15913>`_
and was then offered as a `Google Summer of Code 2022 project <https://openastronomy.org/gsoc/gsoc2022/#/projects?project=gnuastro_library_in_python>`_.
Here, the initial build system and wrappers for some modules of the
``Gnuastro`` Library were built. 

``Gnuastro`` is primarily written in C because
astronomical datasets are large and thus need to be efficient with few
dependencies. Therefore, its most commonly used interface are `Gnuastro’s
command-line programs <https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Command_002dline.html>`_
(that are built on the Unix philosophy). However,
``Gnuastro`` also has an extensive set of installed, dynamic C/C++
libraries, which are the actually beating heart of the programs. On the
other hand, many projects today are done in Python, almost excuslively
using Numpy for their numerical operations (like data arrays). Python and
Numpy are actually written in C, therefore they have very well-defined
interfaces for communicating with installed C libraries.

Thus, pyGnuastro is a low-level wrapper infrastructure which allows easy
usage of Gnuastro's powerful libraries in Python. Also, owing to the
relatively better accessibility of Python, pyGnuastro can act as a gateway
for users/researchers/astronomers who are unfamiliar with programming to
be introduced to ``Gnuastro`` and its Library.
