Code Snippets
=============

The following code snippets demonstrate how to complete common tasks using the
:code:`chromo` package.


Specify Reader Proteins
-----------------------

Default properties for HP1 and PRC1 are implemented; these binders can be called
by name.
New reader proteins can be defined by specifying their physical properties.
After specifying the reader proteins, the `make_binder_collection` method
creates a Pandas table representing the collection of reader proteins, which is
used to define the chromatin during a later step.

.. parsed-literal::

   hp1 = chromo.binders.get_by_name('HP1')
   prc1 = chromo.binders.get_by_name('PRC1')
   custom_binder = ReaderProtein(
      name="new_binder",
      sites_per_bead=2,
      bind_energy_mod=-1,
      bind_energy_no_mod=1,
      interaction_energy=-0.2,
      chemical_potential=-1.5,
      interaction_radius=3
   )
   binders = chromo.binders.make_binder_collection([hp1, prc1, custom_binder])

|

Specify Chemical Modifications & Initial Binding States
-------------------------------------------------------
Each reader protein requires a pattern of histone modifications dictating binding.
These patterns can be specified directly or read from a file.
Initial reader protein binding states may be trivially defined to match the
chemical modifications.
Alternatively, reader proteins may be initialized as unbound.

.. parsed-literal::

   H3K9me3 = Chromatin.load_seqs(["path/to/H3K9me3/sequence"])
   H3K27me3 = Chromatin.load_seqs(["path/to/H3K27me3/sequence"])
   custom_mod = np.zeros((len(H3K9me3))
   chemical_mods = np.column_stack((H3K9me3, H3K27me3, custom_mod))
   states = chemical_mods.copy()
   unbound = np.zeros(chemical_mods.shape, dtype=int)

|

Define Polymer or Chromatin
---------------------------
Homopolymers can be instantiated with basic dimensions and an initial
configuration.
For example, here we define a 1000-beads stretchable, shearable wormlike chain
homopolymer, with beads spaced by 25-units and a persistence length of
100-units.
The polymer is initialized along a Gaussian random walk.

.. parsed-literal::

   num_beads = 1000
   bead_spacing = 25
   lp = 100
   polymer = SSWLC.gaussian_walk_polymer(
      "poly_1",
      num_beads,
      bead_spacing,
      lp=lp
   )

To instantiate chromatin, specify basic dimensions, histone modification
patterns, and an initial configuration.
The polymer below is confined to a sphere with a 900-unit radius.
Each bead is modified at three sites, and initial reader protein binding states
match the initial modifications.

.. parsed-literal::

   chromatin = Chromatin.confined_gaussian_walk(
       'Chr-1',
       num_beads,
       bead_spacing,
       states=states,
       confine_type="Sphere",
       confine_length=900,
       binder_names=np.array(['hp1', 'prc1', 'custom_binder']),
       chemical_mods=chemical_mods,
       chemical_mod_names=np.array(['H3K9me3', 'H3K27me3', 'custom_mod'])
   )

|

Define Uniform Density Field
----------------------------
The density field for the polymer and its binders is instantiated as a grid of
discrete voxels.
The width and number of voxels in each dimension of the field must be specified.

.. parsed-literal::

   x_width = 1000
   y_width = x_width
   z_width = x_width

   n_bins_x = 100
   n_bins_y = n_bins_x
   n_bins_z = n_bins_x

   udf = UniformDensityField(
       polymers = [polymer],
       binders = binders,
       x_width = x_width,
       nx = n_bins_x,
       y_width = y_width,
       ny = n_bins_y,
       z_width = z_width,
       nz = n_bins_z
   )

.. Tip:: Preparation of inputs and analysis of outputs can be completed using
	the :code:`chromo-analysis` package available
	`here <https://github.com/JosephWakim/chromo-analysis>`_.
