<p align="center">
  <img src="https://github.com/MiroK/fenics_ii/blob/master/logo.png">
</p>

(FEniCS)_ii offers some functionality for using FEniCS for PDEs where you combine
equation on one domain with equation on a submanifold embedded in that domain. A
mesh for the embedded manifold does not need to consist of entities of the mesh of the
domain. Crucial for such PDEs is the notion of the trace - hence the name.

## Dependencies
  - [cbc.block](https://bitbucket.org/fenics-apps/cbc.block/src/master/)
  - FEniCS 2019.1.+  (python3)

### Optional dependencies
In some of the demos

  - [HsMG](https://github.com/MiroK/hsmg) is a package implementing preconditioners
    for operators in fractional Sobolev spaces. These are needed to construct
    robust preconditioners for the interface component of the multiscle/multiphysics
  - [ULFY](https://github.com/MiroK/ulfy) is used to construct manufactured
  
## Install
Use pip/distutils or similar. E.g. for dev mode run e.g. `pip install -e .`
or `pip install . --user`, `python3 setup.py install --user`

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
  published. For citing see below.
  
  <p align="center">
    <img src="https://github.com/MiroK/fenics_ii/blob/master/apps/d123/visit0000.png">
  </p>
  
  ## Citing
  If you use FEniCS_ii for your work please cite our [paper](https://link.springer.com/chapter/10.1007/978-3-030-55874-1_63)