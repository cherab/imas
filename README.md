# Cherab IMAS

Cherab add-on module for IMAS (Integrated Modelling & Analysis Suite).

This module enables the creation of Cherab plasma objects and Raysect meshes from IMAS IDS (Interface Data Structures).

## Installation

This add-on module requires IMAS to be installed with Python interface. 

On ITER SDCC load the respective modules:

```bash
module load IMAS
module load Raysect/0.7.1-intel-2020b
```
and then install with:

```bash
pip install cherab==1.4.0 --user
pip install -U cython==3.0a5 --user
pip install <path-to-cherab-imas> --user
```
