# FEniCS_ii DEMOS
Here we collect different PDEs with variables on manifold/having constraints
on manifolds. For each case we strive to give a manufactured solution as well
so that we can verify convergence properties of the FEM formulations (and its
implementation). As an additional test to those in the test suite it is good
idea to lauch `test_demos.py` here.

## Dependencies
For manufactured solutions we rely on [ulfy](https://github.com/MiroK/ulfy) and
direct solvers. For **iterative** demos the preconditioners on the interface are
typically operators in fractional Sobolev spaces implemented in
[HsMG](https://github.com/MiroK/hsmg)

### Coupling 
These demos mostly showcase core functionality of `FEniCS_ii`, that is, the coupling
operators

- [x] `poisson_babuska.py` shows that trace/multiplier meshes need not to conform to the background
- [x] `poisson_babuska_bc.py` illustrates use of boundary conditions
- [x] `mixed_poisson_babuska.py` first encounter with H(div) problem

- [x] `curl_curl_babuska.py` first encounter with H(curl) problem
- [x] `grad_div_babuska.py` sibling problem to `curl_curl.py` (for the sake of completeness)
- [ ] `sym_grad_babuska.py` vector valued problem where we constraint full vector of components on the boundary

- [ ] `stokes_bc.py` nonstandard boundary conditions on Stokes problem; following [Bertoluzza et al](http://dx.doi.org/10.1016/j.cma.2017.04.024)
- [ ] `dq_darcy_stokes.py` first multiphysics problem; Darcy-Stokes formulation without the multiplier following [Discacciati&Quarteroni](https://doi.org/10.1016/S0168-9274(02)00125-3)
- [ ] `layton_darcy_stokes.py` first multiphysics problem with multipliers; Darcy-Stokes formulation without the multiplier following [Layton et al](https://doi.org/10.1137/S0036142901392766)

- [ ] `emi_primal.py`
- [ ] `emi_primal_mortar.py`
- [ ] `twoDoneDoneD.py`

- [x] `poisson_babuska_3d.py` a trace constraint problem in 3d

- [ ] `dq_3d_1d.py`
- [ ] `laurino_3d_1d.py`
- [ ] `lm_3d_1d.py`
- [ ] `isoP2_stokes.py`
- [ ] `point_trace.py`
- [ ] `nonlinear_poisson_babuska.py`
- [ ] `transient_poisson_babuska.py`

### Iterative
We provide robust preconditioners for selected problems (details are provided in the referenced papers)

- [ ] `poisson_babuska_bc_iter.py`
- [ ] `mixed_poisson_babuska_bc_iter.py`
- [ ] `twoDoneDoneD_iter.py`
- [ ] `layton_darcy_stokes_iter.py`