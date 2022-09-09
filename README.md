# Chromo

Physics-based simulator of chromatin architecture.

## Authors
Joseph Wakim, Bruno Beltran, Andrew Spakowitz

## Abstract
Chromatin organization is essential for differential gene expression in humans, enabling a diverse array of cell types to develop from the same genetic code. The architecture of chromatin is dictated by patterns of epigenetic marks–chemical modifications to DNA and histones–which must be reestablished during each cell replication cycle. Dysregulation of chromatin organization has drastic medical consequences and can contribute to aging, obesity, and cancer progression. During this study, we develop a Monte Carlo simulator called “Chromo” to investigate factors affecting chromatin organization. The simulator proposes and evaluates random configurational changes to chromatin, which include geometric transformations and effector protein binding/unbinding to the biopolymer. By iteratively sampling configurations, Chromo generates snapshots of energetically favorable chromatin architectures. Unlike previous models fit to experimental data, Chromo is rooted in fundamental polymer theory, allowing us to predict biophysical mechanisms governing chromatin organization. By leveraging a computationally efficient field theoretic approach, we can simulate full human chromosomes with nucleosome-scale resolution. To confirm the validity of our simulator, we reproduce theoretical chain statistics for flexible and semiflexible homopolymers and recapitulate heterochromatin compartments expected in chromatin contact maps. Using Chromo, we will evaluate how the quality of structural prediction changes with the simulation of additional epigenetic marks. We will also identify which combinations of marks best explain the architecture of the chromosome.

## Quickstart

Our physics-based chromatin simulator is publicly available in the [“Chromo” repository](https://github.com/SpakowitzLab/chromo) of the Spakowitz Lab GitHub account. Clone this repository to your local machine.

`git clone https://github.com/SpakowitzLab/chromo.git`

Install the package dependencies of the simulator, which are listed in requirements.txt; we recommend that you do so in a separate virtual environment using Python 3.9.12. We demonstrate how to do so using the Conda package manager.

`conda create --name chromo python=3.9.12`
`conda activate chromo`
`pip install -r requirements.txt`

Install our simulator package using pip by entering `pip install /path/to/root/directory` in the terminal, specifying the path to the root directory of the codebase on your local machine. If you would like to make changes to the codebase, use `pip install -e /path/to/root/directory` to install the simulator package in editable mode.

`pip install /path/to/root/directory`
`pip install -e /path/to/root/directory`

During installation, all Cython code required by the Monte Carlo algorithm will be compiled. This may take several
minutes. Once the package has been installed, the simulator is ready for use. Navigate to `simulations/examples` from the root directory to find example simulations.


## Code Structure

We use an object-oriented approach to organize components of our model. Components of the system are defined in a hierarchical manner, supporting efficiency and modularity. The class structure of the codebase is illustrated by the diagram below. The accompanying sphinx documentation includes descriptions of each class and their attributes.

<img src="doc/source/figures/chromo_UML.png">

## Examples
We've prepared example simulations for a variety of polymer modeling scenarios, which are presented as jupyter notebooks at the `simulations/examples` subdirectory.
