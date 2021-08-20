# FEniCS_ii DEMOS
Here we collect different PDEs with variables on manifold/having constraints
on manifolds. For each case we strive to give a manufactured solution as well
so that we can verify convergence properties of the FEM formulations (and its
implementation). As an additional test to those in the test suite it is good
idea to lauch `test_demos.py`/`py.test .` from this folder.

## Dependencies
For manufactured solutions we rely on [ulfy](https://github.com/MiroK/ulfy) and
direct solvers. For **iterative** demos the preconditioners on the interface are
typically operators in fractional Sobolev spaces implemented in
[HsMG](https://github.com/MiroK/hsmg). Demo `stokes_bc.py` can be run with a flag
`--is_flat 0` in which case domain with curved boundary is considered. We generate
mesh for it on the fly if [gmshnics](https://github.com/MiroK/gmshnics) is installed.

### Coupling 
These demos mostly showcase core functionality of `FEniCS_ii`, that is, the coupling
operators. In mixed-dimensional systems a commonly used reduction operator is the trace
resctricting the function to manifold of codimension 1.

- [x] `poisson_babuska.py` shows that trace/multiplier meshes need not to conform to the background
- [x] `poisson_babuska_bc.py` illustrates use of boundary conditions
- [x] `mixed_poisson_babuska.py` first encounter with H(div) problem
- [x] `curl_curl_babuska.py` first encounter with H(curl) problem
- [x] `grad_div_babuska.py` sibling problem to `curl_curl.py` (for the sake of completeness)
- [x] `sym_grad_babuska.py` vector valued problem where we constrain full vector of components on the boundary
- [x] `stokes_bc.py` nonstandard boundary conditions on Stokes problem; following [Bertoluzza et al](http://dx.doi.org/10.1016/j.cma.2017.04.024)
- [x] `poisson_babuska_3d.py` a trace constraint problem in 3d

We include some systems related to (reduced) EMI models, where variable-coefficient diffusion
equations posed on domains of same/different topological dimension are coupled.

- [x] `emi_primal.py` formulation EMI model without the multiplier discussed e.g. in [Kuchta et al](https://doi.org/10.1007/978-3-030-61157-6_5)
- [x] `emi_primal_mortar.py` original EMI formulation with Lagrange multiplier presented in [Tveito et al](https://doi.org/10.3389/fphy.2017.00048)
- [x] `twoDoneDoneD.py` coupling between 2d-1d and 1d domains; [system](https://doi.org/10.1137/15M1052822) obtained by dimensional reduction from mortar EMI

Keeping the trace coupling we now consider different physics on different bulk domains

- [x] `dq_darcy_stokes.py` first multiphysics problem; Darcy-Stokes formulation without the multiplier following [Discacciati&Quarteroni](https://doi.org/10.1016/S0168-9274(02)00125-3)
- [x] `layton_darcy_stokes.py` first multiphysics problem with multipliers; Darcy-Stokes formulation without the multiplier following [Layton et al](https://doi.org/10.1137/S0036142901392766)

An extra feature of `FEniCS_ii` is coupling which bridges the topological gap larger
than 1, in particular, we support 3d to 1d restriction by averaging over virtual surfaces
(in addition to mathematically less sound 3d to 1d trace). 

- [ ] `dq_3d_1d.py`
- [ ] `laurino_3d_1d.py`
- [ ] `lm_3d_1d.py`

Some other restriction operators provided are e.g. `Injection`, `PointTrace`

- [x] `isoP2_stokes.py` stabilizing P1-(P1|P0) elements by considering coarser mesh for the pressure,
see [Bercovier&Pironneau](https://doi.org/10.1007/BF01399555) or Chen Long's [experiments](https://www.math.uci.edu/~chenlong/ifemdoc/fem/StokesisoP2P0femrate.html)
- [ ] `point_trace.py`

## Extras
Some experimental extensions to transient or nonlinear problems are show cased in

- [ ] `nonlinear_poisson_babuska.py`
- [ ] `transient_poisson_babuska.py`

### Iterative
We provide robust preconditioners for selected problems (details are provided in the referenced papers)

- [x] `poisson_babuska_bc_iter.py` preconditioner based on well-posedness established in [Babuska](https://doi.org/10.1007/BF01436561)
- [x] `mixed_poisson_babuska_bc_iter.py` preconditioner based on well-posedness established in [Babuska&Gatica](https://doi.org/10.1002/num.10040)
- [x] `twoDoneDoneD_iter.py` precondiner based on analysis in [Kuchta et al](https://doi.org/10.1137/15M1052822)
- [x] `layton_darcy_stokes_iter.py` intersection space preconditioner from [Holter et al](https://arxiv.org/abs/2001.05527)