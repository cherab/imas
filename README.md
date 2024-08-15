# Cherab IMAS

Cherab add-on module for IMAS (Integrated Modelling & Analysis Suite).

This module enables the creation of Cherab plasma objects and Raysect meshes from IMAS IDS (Interface Data Structures).

## Installation

This add-on module requires IMAS to be installed with Python interface. 

Imnstall with the following commands.

On ITER SDCC CentOS8 nodes:

```bash
module load IMAS
module load Raysect/0.7.1-intel-2020b
pip install <path-to-cherab-imas> --user
```

On ITER SDCC RHEL9 nodes:

```bash
module load IMAS
module load Raysect/0.7.1-iimkl-2023b
pip install <path-to-cherab-imas> --user
```
