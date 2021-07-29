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
- [ ] `poisson_babuska_bc.py` illustrates use of boundary conditions
- [ ] `mixed_poisson_babuska.py` first encounter with H(div) problem
- [ ] `curl_curl_babuska.py` first encounter with H(curl) problem
- [ ] `grad_div_babuska.py` sibling problem to `curl_curl.py`
- [ ] `stokes_bc.py` nonstandard boundary conditions on Stokes problem; following Bertoluzza et al.
- [ ] `dq_darcy_stokes` first multiphysics problem; Darcy-Stokes formulation without the multiplier following Discacciati&Quarteroni
- [ ] `layton_darcy_stokes.py` first multiphysics problem with multipliers; Darcy-Stokes formulation without the multiplier following Layton
- [ ] `emi_primal.py`
- [ ] `emi_primal_mortar.py`
- [ ] `twoDoneDoneD.py`
- [ ] `dq_3d_1d.py`
- [ ] `laurino_3d_1d.py`
- [ ] `lm_3d_1d.py`
- [ ] `isoP2_stokes.py`
- [ ] `point_trace.py`
- [ ] `nonlinear_poisson_babuska.py`

### Iterative
We provide robust preconditioners for selected problems (details are provided
in the referenced papers)

- [ ] `poisson_babuska_bc_iter.py`
- [ ] `mixed_poisson_babuska_bc_iter.py`
- [ ] `twoDoneDoneD_iter.py`
- [ ] `layton_darcy_stokes_iter.py`