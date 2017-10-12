<p align="center">
  <img src="https://github.com/MiroK/fenics_ii/blob/master/logo.png">
</p>

(FEniCS)_ii offers some functionality for using FEniCS for PDEs where you combine
equation on one domain with equation on a submanifold embedded in that domain. A
mesh for the embedded manifold must consist of entities of the mesh of the
domain. Crucial for such PDEs is the notion of the trace - hence the name.

Dependencies:
  - cbc.block
  - FEniCS stack prior to
    DOLFIN's [commit](https://bitbucket.org/fenics-project/dolfin/commits/670c1f385c27d5ce64e9123114baa33f945761a4) which introduced this [bug](https://bitbucket.org/fenics-project/dolfin/issues/805/dirichletbc-check_arguments-is-too-strict)

To install the package put this directory on python path. For this shell
session you can achieve this by `source setup.rc`. 
