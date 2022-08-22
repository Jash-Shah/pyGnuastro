import pygnuastro.cosmology

class TestCosmology:
  def test_age(self):
    assert pygnuastro.cosmology.age(z=5, H0=60, 
                                    olambda=0.2, omatter=0.3,
                                    oradiation=0.5) == 0.3101850380399441
  
  def test_age_default(self):
    assert pygnuastro.cosmology.age(3) == 2.1483618980550068

  def test_velocity_from_z(self):
    assert pygnuastro.velocity_from_z(5) == 283587.4602702703