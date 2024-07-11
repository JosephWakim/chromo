.. _field:


Field-Theoretic Treatment of Interactions
=========================================

Our treatment of interactions uses a field-theoretic treatment of the densities to determine the interactions between polymer
segments.  Following work by Pike, et al. (Refs. [Pike2009a]_, [Pike2009b]_),
we define

The simulation has a fixed volume with sides lengths :math:`L_{x}`, :math:`L_{y}`, and :math:`L_{z}`.
These lengths are discretize into :math:`M_{x}`, :math:`M_{y}`, and :math:`M_{z}` bins of length
:math:`\Delta_{x} = L_{x}/M_{x}`,
:math:`\Delta_{y} = L_{y}/M_{y}`, and
:math:`\Delta_{z} = L_{z}/M_{z}`.
The bins are defined by the three indices
:math:`i_{x}`,
:math:`i_{y}`, and
:math:`i_{z}` that run from zero to
:math:`M_{x}-1`,
:math:`M_{y}-1`, and
:math:`M_{z}-1`, respectively.


We consider the :math:`n` th bead located at position :math:`\vec{r}^{(n)}`.
We define a weight function :math:`w_{I}(\vec{r}^{(n)})` within the :math:`I`th bin.
The :math:`I`th index is defined to be a superindex that combines
:math:`i_{x}`,
:math:`i_{y}`, and
:math:`i_{z}` into a single unique index :math:`I= i_{x} + M_{x} i_{y} + M_{x}M_{z} i_{z}` that
runs from zero to :math:`M_{x}M_{y}M_{z}-1` (total of :math:`M_{x}M_{y}M_{z}` unique indices)
The total weight on the :math:`I`th bin is given by the contributions from the three cartesian
directions, `i.e.`
:math:`w_{I}(\vec{r}^{(n)}) =
w_{i_{x}}^{(x)}(x^{(n)})
w_{i_{y}}^{(y)}(y^{(n)})
w_{i_{z}}^{(z)}(z^{(n)})`.
The figure below shows a schematic of the :math:`x`-direction weight function (same method for :math:`y` and :math:`z`).
This shows a linear interpolation weighting method, consistent with Refs. [Pike2009a]_, [Pike2009b]_.

.. figure:: figures/weight.pdf
    :width: 600
    :align: center
    :alt: Schematic of the weight function :math:`w_{i_{x}}^{(x)}` that gives the weighting of the particle in the :math:`i_{x}` site in the
        :math:`x`-direction based on a linear interpolation method

    Schematic of the weight function :math:`w_{i_{x}}^{(x)}` that gives the weighting of the particle in the :math:`i_{x}` site in the
    :math:`x`-direction based on a linear interpolation method


The number of epigenetic proteins (`e.g.` HP1) to the :math:`n`th site is given by :math:`N_{I}^{(\alpha)}`, where :math:`\alpha` determines
the type of epigenetic mark.
The :math:`\alpha`-protein density within the :math:`I`th bin is given by

.. math::
    \rho_{I}^{(\alpha)} = \frac{1}{v_{\mathrm{bin}}} \sum_{n=0}^{n_{b} - 1} w_{I}(\vec{r}^{(n)}) N_{I}^{(\alpha)}

where :math:`v_{\mathrm{bin}} = \Delta_{x} \Delta_{y} \Delta_{z}` is the volume of a bin.
The maximum number of epigenetic proteins bound :math:`N_{\mathrm{max}}^{(\alpha)}` gives an upper bound on the
number of proteins that can bind to a bead, accounting for coarse graining of a bead to represent multiple nucleosomes.
For discretization of one nucleosome per bead, the maximum :math:`N_{\mathrm{max}}^{(\alpha)} = 2` implies binding
of a protein to the two histone tail proteins for the :math:`\alpha` epigenetic mark.
We define the number of :math:`\alpha` marks on the :math:`I`th bead as :math:`M_{I}^{(\alpha)}`, which can take values from zero
to :math:`N_{\mathrm{max}}^{(\alpha)}`.

Protein binding to a marked tail results in energy :math:`-\beta \epsilon_{m}` [non-dimensionalized by :math:`\beta = 1/(k_{B}T)`], and protein binding to an unmarked tail is associated with
energy :math:`-\beta \epsilon_{u}`.  The chemical potential of the :math:`\alpha` protein is defined as :math:`\beta \mu^{(\alpha)}`.
The binding of :math:`N_{I}^{(\alpha)}` proteins to a bead with :math:`M_{I}^{(\alpha)}` marks results in a free energy that
accounts for all of the combinatoric ways of binding.
