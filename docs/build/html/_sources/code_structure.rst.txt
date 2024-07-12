.. _code_structure:

Code Overview and Structure
===========================

Chromo is implemented with an object-oriented approach to accommodate flexible polymer models at various scales, bound by arbitrary numbers of interacting reader proteins.
The simulator is specifically designed to model chromatin, but can be adapted to model arbitrary block co-polymers with or without confinement.

The hierarchcial organization of chromatin (and block copolymers in general) is reflected by the simulator's class structure.
At the largest scale, instances of the :code:`Mixture` class contain all interacting polymers in the system.
Each chromatin strand is instantiated as a :code:`Chromatin` object, which adopts physical properties of a stretchable, shearable wormlike chain.
:code:`Chromatin` instances track a sequence of nucleosomes, each of which is instantiated as a :code:`Nucleosome` object.
The :code:`Chromatin` class also stores patterns of chemical modifications and associated reader protein binding states.
Physical properties of the reader proteins, which define how reader proteins interact, are specified as attributes of the :code:`ReaderProtein` class.
An instance of the :code:`UniformDensityField` class maintains a discrete density field for the polymer and all its reader proteins.
This field is used to efficiently evaluate the interaction energy of components in the simulation.

To enable convenient adaption of our software for different polymer models, we defined abstract classes for each component of our system, and we leveraged a hierarchy of subclasses with increasing levels of detail.
For example, the :code:`Chromatin` class inherits from a :code:`SSWLC` class that defines physical properties of a stretchable, shearable wormlike chains, including methods for evaluating elastic energy.
The :code:`SSWLC` class inherits from a :code:`PolymerBase` class that includes required attributes and bookkeeping methods for any polymer.
Below is a schematic of hierarchical organization of our simulator.

.. image:: figures/chromo_UML.png

|

The software is written in Python, providing interpretable code and access to a broad library of published packages.
However, as an interpreted language, Python code is notoriously slow to run.
To work around this limitation, parts of the codebase were written in "Cython," a variation of Python that allows for direct translation to C code.
All parts of the codebase involved in the inner-loop of the Monte Carlo algorithm have been translated to Cython.