# UMEP Core

## Setup

- Make sure you have a Python installation on your system
- Install `vscode` and `github` apps.
- Install `uv` package manager (e.g. `pip install uv`).
- Clone repo.
- Run `uv sync` from the directory where `pyproject.toml` in located to install `.venv` and packages.
- Select `.venv` Python environment.
- FYI: Recommended settings and extensions are included in the repo. Proceed if prompted to install extensions.
- Develop and commit to Github often!

## Demo

See the demo notebook file at [/demo.py](/demo.py).

Also, a test with GBG data is found in [/solweig_gbg_test.py](/solweig_gbg_test.py)

The demo and the test uses the datasets included in the tests folder

## Original code

The code reproduced in the `umep` folder is adapted from the original GPLv3-licensed code by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, Nils Wallenberg.

The original code has been modified to work without QGIS to facilitate Python workflows.

The original code can be found at: [UMEP-processing](https://github.com/UMEP-dev/UMEP-processing).

This modified code is licensed under the GNU General Public License v3.0.

See the LICENSE file for details.

Please give all credit for UMEP code to the original authors and cite accordingly.

© Copyright 2018 - 2020, Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, Nils Wallenberg.

Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel- Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software.99, 70-87 https://doi.org/10.1016/j.envsoft.2017.09.020

## Demo Data

Two seprated demo dataset are included

### ATENS (vector data)

#### Tree Canopies

Copernicus

#### Trees

https://walkable.cityofathens.gr/home

#### Buildings

http://gis.cityofathens.gr/layers/athens_geonode_data:geonode:c40solarmap

### Gothenburg (raster data)

Standard dataset used in tutorials (https://umep-docs.readthedocs.io/en/latest/Tutorials.html)

