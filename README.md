# Cherab IMAS

Cherab add-on module for IMAS (Integrated Modelling & Analysis Suite).

This module enables the creation of Cherab plasma objects and Raysect meshes from IMAS IDS (Interface Data Structures).

## Installation

This add-on module requires IMAS to be installed with Python interface. 

On ITER SDCC load the IMAS module:

```bash
module load IMAS
```
and then install with:

```bash
pip install git@https://github.com/cherab/imas --user
```
Alternatively, install to a virtual environment:

```bash
python -m venv /path/to/cherab_virtual_environment
source /path/to/cherab_virtual_environment/bin/activate
pip install git@https://github.com/cherab/imas
```
