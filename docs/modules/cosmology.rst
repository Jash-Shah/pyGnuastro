************************************
Cosmology (``pygnuastro.cosmology``)
************************************

This library does the main cosmological calculations that are commonly
necessary in extra-galactic astronomical studies. The main variable in
this context is the redshift (``z``). The cosmological input parameters in
the functions below are ``H0, o_lambda_0, o_matter_0, o_radiation_0`` which
respectively represent the current (at redshift 0) expansion rate (Hubble
constant in units of ``km/sec/Mpc``), cosmological constant (``Λ``), matter
and radiation densities. The ``pygnuastro.cosmology`` module provides the
same functions as it’s `Gnuastro Library counterpart
<https://www.gnu.org/savannah-checkouts/gnu/gnuastro/manual/html_node/Cosmology-library.html>`_.

.. note::

  For all functions below, the redshift(``z``) is the only required argument. The default values
  taken by the other arguments are: ``H0 = 67.66, olambda = 0.6889, omatter = 0.3111,
  oradiation = 0.0``. (Constants from `Plank 2018 (arXiv:1807.06209, Table 2)
  <https://arxiv.org/abs/1807.06209>`_)
  As the summation of the cosmological constants(``olambda, omatter`` and ``oradiation``)
  should be 1, in order to pass custom values for these, all three have to be provided by the
  user.

.. autofunction:: pygnuastro.cosmology.age
.. autofunction:: pygnuastro.cosmology.angular_distance
.. autofunction:: pygnuastro.cosmology.to_absolute_mag
.. autofunction:: pygnuastro.cosmology.critical_density
.. autofunction:: pygnuastro.cosmology.comoving_volume
.. autofunction:: pygnuastro.cosmology.proper_distance
.. autofunction:: pygnuastro.cosmology.luminosity_distance
.. autofunction:: pygnuastro.cosmology.distance_modulus
.. autofunction:: pygnuastro.cosmology.velocity_from_z
.. autofunction:: pygnuastro.cosmology.z_from_velocity