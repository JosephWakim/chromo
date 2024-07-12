chromo package
==============

This package contains the main components of the polymer system being simulated.
Each module represents a different scale of chromatin.
At the largest scale are **fields**, which are implicit representations of
polymers and their interactions. Taking a field theoretic approach, we can
efficiently simulate whole chromosomes at nucleosome-scale resolution.
Next, the **mixture** module deals with explicit interactions between beads on
multiple polymer chains.
The **polymer** module provides classes for different polymer modles and methods
for evaluating elastic energy.
The **beads** module provides classes to represent the discrete monomeric units
of the polymer with various levels of detail.
Finally, at the smallest scale, the **binders** module provides chemical
properties of proteins or other bound material that dictate their interaction
with the polymer and with each other.

chromo.fields
-------------

.. automodule:: chromo.fields
   :members:
   :show-inheritance:

chromo.mixtures
---------------

.. automodule:: chromo.mixtures
   :members:
   :show-inheritance:

chromo.polymers
---------------

.. automodule:: chromo.polymers
   :members:
   :show-inheritance:

chromo.beads
------------

.. automodule:: chromo.beads
   :members:
   :show-inheritance:

chromo.binders
--------------

.. automodule:: chromo.binders
   :members:
   :show-inheritance:

chromo.__init__
---------------

.. automodule:: chromo
   :members:
   :show-inheritance:
