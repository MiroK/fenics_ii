<p align="center">
  <img src="https://github.com/MiroK/fenics_ii/blob/master/logo.png">
</p>

(FEniCS)_ii offers some functionality for using FEniCS for PDEs where you combine
equation on one domain with equation on a submanifold embedded in that domain. A
mesh for the embedded manifold must consist of entities of the mesh of the
domain. Crucial for such PDEs is the notion of the trace - hence the name.

## Dependencies
  - cbc.block
  - FEniCS 2017.2.0  (python2)

## Install
To install the package put this directory on python path. For this shell
session you can achieve this by `source setup.rc`. 

## Limitations
 - Trace(expr) where expr is not a UFL terminal isn't currently supported
 - Point constraints
 - MPI parallelism
 
 ## What it can do
  Coupled $X$d-$Y$d, $X >=Y >= 1$ and 3d-2d-1d coupled problems where the trace of 
  terminal (see limitations) is used in the coupling. See [demos](https://github.com/MiroK/fenics_ii/tree/master/demo) 
  and [apps](https://github.com/MiroK/fenics_ii/tree/master/apps) folders for 
  examples. Please note that the demos/apps often include mesh independent 
  preconditioners or nontrivial physics problem formulations which the authors devolop as 
  part of their research. Much like this code they are in various stages of getting/being
  published. Please contact us about citing and credits.
  
  <p align="center">
    <img src="https://github.com/MiroK/fenics_ii/blob/master/apps/d123/visit0000.png">
  </p>
  
