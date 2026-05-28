# p-wave magnet lattice Hamiltonians and lesser Green's functions

Author: Sigri M. Sveen

## Description

This code implements a tight-binding non-equilibrium Green's
function (NEGF) framework for constructing p-wave magnet lattice
Hamiltonians and calculating lesser Green's functions.

The system consists of a central square-lattice region coupled to
four semi-infinite leads situated to the left, right, above, and
below the central region. Each lead is connected at infinity to
a reservoir fixed to a given Fermi level and temperature.

The resulting Green's functions may subsequently be used to
evaluate quantum transport quantities through energy integrations
and visualized, for instance, through quiver plots.

## Implemented Hamiltonian models

* Itinerant non-collinear p-wave model
* Localized helimagnetic non-collinear p-wave model
* Collinear p-wave model

## Main capabilities

* Construction of p-wave magnet lattice Hamiltonians
* Lead self-energy calculations
* Calculation of the system's lesser Green's functions

## Installation

This code requires only NumPy:

```bash
pip install numpy
```
