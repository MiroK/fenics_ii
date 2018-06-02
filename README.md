<p align="center">
  <img src="https://github.com/MiroK/fenics_ii/blob/master/logo.png">
</p>

(FEniCS)_ii offers some functionality for using FEniCS for PDEs where you combine
equation on one domain with equation on a submanifold embedded in that domain. A
mesh for the embedded manifold must consist of entities of the mesh of the
domain. Crucial for such PDEs is the notion of the trace - hence the name.

Dependencies:
  - cbc.block
  - FEniCS 2017.2.0  (python2)

To install the package put this directory on python path. For this shell
session you can achieve this by `source setup.rc`. 

[[https://github.com/MiroK/fenics_ii/blob/master/apps/d123/visit0000.png]]

Limitations:
 - Trace(expr) where expr is not a UFL terminal isn't currently supported
 - Point constraints
 - MPI parallelism
 
