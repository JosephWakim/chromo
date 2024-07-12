.. _poly_models:

Polymer Models
==============

We introduce the polymer models that can be used in the chromo simulation.
These models are designed to capture different physical effects.
They are all based on a common representation of the
chain as a discrete set of beads.
However, each model uses a subset of the geometric variables in their
calculation of energy, force, and torque, and the simulation automatically
adjusts the underlying calculations based on the model.
Here, we define the general chain representation that includes the most
complete geometric variables for the chain, and each polymer model utilizes
a subset of these variables (defined for each model below).
Each model also has the option of creating a linear or ring polymer,
which simply requires the model to add a bond between the first bead and the
last bead of the chain.

We consider a polymer with :math:`n_{b}` number of beads in a single chain.
The polymer chain is represented by the
bead positions
:math:`\vec{r}^{(0)}, \vec{r}^{(1)}, \ldots, \vec{r}^{(n_{b}-1)}`
and orientations
:math:`\vec{t}_{i}^{(0)}, \vec{t}_{i}^{(1)}, \ldots, \vec{t}_{i}^{(n_{b}-1)}`
where :math:`i = 1, 2, 3`.
Since the orientation triad forms an orthonormal basis, the dynamics must maintain the following conditions:

.. math::
    \vec{t}_{i}^{(n)} \cdot \vec{t}_{j}^{(n)} = \delta_{ij}

Each model represents a chain whose total length is :math:`L`.
For linear polymers, the chain is discretized into segments of length
:math:`\Delta = L/(n_{b}-1)`. This implies that the end-to-end
vector is given by :math:`\vec{R} = \vec{r}^{(n_{b}-1)} - \vec{r}^{(0)}`.
For ring polymers, the chain is discretized into segments of length
:math:`\Delta = L/n_{b}`, which includes the bond between the last
bead at :math:`\vec{r}^{(n_{b} - 1)}` and the first bead at
:math:`\vec{r}^{(0)}`.

|

Polymer chain model A: flexible Gaussian chain
----------------------------------------------

The flexible Gaussian chain captures the behavior of a flexible polymer chain that
is representative of a Gaussian random walk in the absence of additional interactions.
The chain is defined by the bead positions :math:`\vec{r}^{(n)}`, and the
bead orientations are not utilized in this model.

We define the polymer energy function

.. math::
    \beta E_{\mathrm{poly}} = \sum_{n=0}^{n_{b}-2}
    \frac{3}{2 \Delta b} \left( \vec{r}^{(n+1)} - \vec{r}^{(n)} \right)^{2}

where we define :math:`\beta = 1/(k_{B}T)`, and the Kuhn length
:math:`b` defines the statistical segment length of the polymer chain.
This energy definition is valid for a linear chain.
However, the ring representation only requires the upper limit of the
summation to be changed to :math:`n_{b} - 1`, and we note that
the condition :math:`\vec{r}^{(n_{b})} = \vec{r}^{(0)}` connects the
chain ends into a ring.
In this model, the bead orientations :math:`\vec{t}_{i}^{(n)}` do not
contribute to the energy and are not evolved in the simulation.

From the polymer energy, we determine the force on the nth bead
is determined from :math:`\vec{f}^{(n)} = - \frac{\partial \beta E}{\partial \vec{r}^{(n)}}`,
where we scale the force by the thermal energy :math:`k_{B}T`.
This results in the force on the nth bead to be given as

.. math::
    \vec{f}^{(n)} = \frac{3}{\Delta b} \left( \vec{r}^{(n+1)}
    - 2 \vec{r}^{(n)} + \vec{r}^{(n-1)} \right)

for all beads within the interior of the chain.
This expression also applies to the end beads :math:`n=0` and
:math:`n=n_{b}-1` for a ring polymer, if we note that
:math:`\vec{r}^{(-1)} = \vec{r}^{(n_{b}-1)}` and
:math:`\vec{r}^{(n_{b})} = \vec{r}^{(0)}`.

For the ends of a linear chain, the end-bead forces are given by

.. math::
    \vec{f}^{(0)} & = & \frac{3}{\Delta b} \left(
    \vec{r}^{(1)}
    - \vec{r}^{(0)}
    \right) \\
    \vec{f}^{(n_{b}-1)} & = & - \frac{3}{\Delta b} \left(
    \vec{r}^{(n_{b}-1)}
    - \vec{r}^{(n_{b}-2)}
    \right)

Initialization of the flexible Gaussian chain in the absence of additional interactions
(e.g. excluded-volume interactions or confinement)
is performed by selecting the bond vectors from a Gaussian distribution with
a variance that is given by

.. math::
    \langle
    \left(
    \vec{r}^{(n+1)} - \vec{r}^{(n)}
    \right)
    \left(
    \vec{r}^{(n'+1)} - \vec{r}^{(n')}
    \right)
    \rangle = \frac{\Delta b}{3} \delta_{nn'} \mathbf{I}

which leads to a mean-square end-to-end distance

.. math::
    \langle
    \vec{R}^{2}
    \rangle =
    \sum_{n=0}^{n_{b}-2}
    \sum_{n'=0}^{n_{b}-2}
    \langle
    \left(
    \vec{r}^{(n+1)} - \vec{r}^{(n)}
    \right) \cdot
    \left(
    \vec{r}^{(n'+1)} - \vec{r}^{(n')}
    \right)
    \rangle
    = \sum_{n=0}^{n_{b}-2}
    \sum_{n'=0}^{n_{b}-2}
    \Delta b \delta_{nn'}
    = \Delta b (n_{b} - 1)
    = L b

which is consistent with the solution for a Gaussian random
walk polymer with length :math:`L = N b`,
where :math:`N` is the number of Kuhn lengths in the
chain.

|

Polymer chain model B: shearable, stretchable wormlike chain
------------------------------------------------------------

We consider the shearable, stretchable wormlike chain potential, given by

.. math::
    \beta E_{\mathrm{poly}} = \sum_{n=0}^{n_{b}-2}
    \left[
    \frac{\epsilon_{\mathrm{b}}}{2 \Delta} \left| \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{\parallel}}}{2 \Delta} \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right)^{2} +
    \frac{\epsilon_{\mathrm{\perp}}}{2 \Delta} \left| \Delta \vec{r}_{\perp}^{(n)} \right|^{2}
    \right],

where :math:`\Delta \vec{r}^{(n)} = \vec{r}^{(n+1)} - \vec{r}^{(n)}` is the bond vector,
:math:`\Delta \vec{r}_{\perp}^{(n)} = \Delta \vec{r}^{(n)} - (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{t}_{3}^{(n)}`
is the perpendicular component of the bond vector to the tangent vector.

|

Polymer chain model C: shearable, stretchable wormlike chain with twist
-----------------------------------------------------------------------

We consider a closed ring polymer, where the :math:`n_{b}` bead is the same as the zeroth bead.
We consider the shearable, stretchable wormlike chain potential with twist, given by

.. math::
    \beta E_{\mathrm{poly}} = \sum_{n=0}^{n_{b}-1}
    \left[
    \frac{\epsilon_{\mathrm{b}}}{2 \Delta} \left| \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{\parallel}}}{2 \Delta} \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right)^{2} +
    \frac{\epsilon_{\mathrm{\perp}}}{2 \Delta} \left| \Delta \vec{r}_{\perp}^{(n)} \right|^{2} +
    \frac{\epsilon_{\mathrm{t}}}{2 \Delta} \left( \omega^{(n)} \right)^{2}
    \right],

where :math:`\Delta \vec{r}^{(n)} = \vec{r}^{(n+1)} - \vec{r}^{(n)}` is the bond vector,
:math:`\Delta \vec{r}_{\perp}^{(n)} = \Delta \vec{r}^{(n)} - (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{t}_{3}^{(n)}` is the
perpendicular component of the bond vector to the tangent vector.

The twist angle :math:`\omega^{(n)}` gives the
local twist deformation of the chain.
Geometrically, this is defined by the relationship

.. math::
    \left( 1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} \right) \cos \Omega^{(n)} & = &
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}  \\
    \left( 1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} \right) \sin \Omega^{(n)} & = &
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)} -
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)}

where :math:`\omega^{(n)} = \Omega^{(n)} + 2 \pi m^{(n)}`,
and :math:`m^{(n)}` gives the number of additional integer turns
of twist within the
nth segment.
We write a differential change in :math:`\omega^{(n)}` as

.. math::
    \delta \omega^{(n)} & = &
    \frac{\vec{t}_{1}^{(n+1)} \cdot \delta \vec{t}_{2}^{(n)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    } -
    \frac{\vec{t}_{2}^{(n+1)} \cdot \delta \vec{t}_{1}^{(n)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }
    \nonumber \\
    &  &
    + \frac{\vec{t}_{2}^{(n)} \cdot \delta \vec{t}_{1}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }  -
    \frac{\vec{t}_{1}^{(n)} \cdot \delta \vec{t}_{2}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}
    }  \nonumber \\
    &  &
    -  \left(
    \frac{\vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)}  - \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    {\vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    \right)
    \left(
    \frac{
    \vec{t}_{3}^{(n)} \cdot \delta \vec{t}_{3}^{(n+1)} +
    \vec{t}_{3}^{(n+1)} \cdot \delta \vec{t}_{3}^{(n)}
    }{1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} }
    \right)


With this development, we write the torque vectors as

.. math::
    \vec{\tau}_{1}^{(n)} & = & \frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{2}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}}
    \right)
    -
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{2}^{(n-1)}}{
    \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)}}
    \right)
    \\
    \vec{\tau}_{2}^{(n)} & = & - \frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{1}^{(n+1)}}{
    \vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)}}
    \right)
    +
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{1}^{(n-1)} }{
    \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)}}
    \right) \\
    \vec{\tau}_{3}^{(n)} & = &
    \vec{\tau}_{b}^{(n)} -
    \vec{\tau}_{b}^{(n-1)} - \eta \left[
    (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)}) \vec{\tau}_{b}^{(n)}
    + ( \vec{\tau}_{b}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \Delta \vec{r}^{(n)}
    \right]
    \nonumber \\
    &  &
    - \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right) \Delta \vec{r}^{(n)}
    + \frac{\epsilon_{\perp}}{\Delta}
    (\Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \Delta \vec{r}_{\perp}^{(n)}
    \nonumber \\
    &  &
    +\frac{\epsilon_{t}}{\Delta} \omega^{(n)}
    \left(
    \frac{\vec{t}_{2}^{(n)} \cdot \vec{t}_{1}^{(n+1)}  - \vec{t}_{1}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    {\vec{t}_{1}^{(n)} \cdot \vec{t}_{1}^{(n+1)} +
    \vec{t}_{2}^{(n)} \cdot \vec{t}_{2}^{(n+1)} }
    \right)
    \frac{
    \vec{t}_{3}^{(n+1)}}{1 + \vec{t}_{3}^{(n)} \cdot \vec{t}_{3}^{(n+1)} } \nonumber \\
    &  &
    +
    \frac{\epsilon_{t}}{\Delta} \omega^{(n-1)}
    \left(
    \frac{\vec{t}_{2}^{(n-1)} \cdot \vec{t}_{1}^{(n)}  - \vec{t}_{1}^{(n-1)} \cdot \vec{t}_{2}^{(n)} }
    {\vec{t}_{1}^{(n-1)} \cdot \vec{t}_{1}^{(n)} +
    \vec{t}_{2}^{(n-1)} \cdot \vec{t}_{2}^{(n)} }
    \right)
    \frac{
    \vec{t}_{3}^{(n-1)}}{1 + \vec{t}_{3}^{(n-1)} \cdot \vec{t}_{3}^{(n)} }

where

.. math::
    \vec{\tau}_{b}^{(n)} =
    \frac{\epsilon_{b}}{\Delta} \left(
    \vec{t}_{3}^{(n+1)} - \vec{t}_{3}^{(n)} - \eta \Delta \vec{r}_{\perp}^{(n)}
    \right)

The force on the nth bead is given by

.. math::
    \vec{f}^{(n)} & = &
    -\eta \vec{\tau}_{b}^{(n)} + \eta ( \vec{\tau}_{b}^{(n)} \cdot \vec{t}_{3}^{(n)} ) \vec{t}_{3}^{(n)}
    +\eta \vec{\tau}_{b}^{(n-1)} - \eta ( \vec{\tau}_{b}^{(n-1)} \cdot \vec{t}_{3}^{(n-1)} ) \vec{t}_{3}^{(n-1)}
    \nonumber \\
    &  &
    + \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n)} \cdot \vec{t}_{3}^{(n)} - \Delta \gamma \right) \vec{t}_{3}^{(n)}
    - \frac{\epsilon_{\parallel}}{\Delta}
    \left( \Delta \vec{r}^{(n-1)} \cdot \vec{t}_{3}^{(n-1)} - \Delta \gamma \right) \vec{t}_{3}^{(n-1)}
    \nonumber \\
    &  &
    + \frac{\epsilon_{\perp}}{\Delta}
    \Delta \vec{r}_{\perp}^{(n)}
    - \frac{\epsilon_{\perp}}{\Delta}
    \Delta \vec{r}_{\perp}^{(n-1)}

