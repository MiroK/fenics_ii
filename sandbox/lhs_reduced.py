from block import block_mat, block_vec, block_bc
from dolfin import *
from xii import *

import mshr

N = 10
EPS = 1E-3
R = 0.25
box_domain = mshr.Box(dolfin.Point(0, 0, 0), dolfin.Point(1, 1, 1))
_mesh = mshr.generate_mesh(box_domain, N)

stokes_subdomain = dolfin.CompiledSubDomain(
    "sqrt((x[0]-0.5) * (x[0]-0.5) + (x[1]-0.5) * (x[1]-0.5)) < R", R=R
)

subdomains = MeshFunction('size_t', _mesh, _mesh.topology().dim(), 0)

# Awkward marking
for cell in cells(_mesh):
    x = cell.midpoint().array()
    if stokes_subdomain.inside(x, False):
        subdomains[cell] = 1
    else:
        subdomains[cell] = 0

submeshes, interface, _ = mortar_meshes(subdomains, range(2), strict=True, tol=EPS)


stokes_domain = submeshes[0]
porous_domain = submeshes[1]
interface = interface

## function spaces

# biot
Vp = FunctionSpace(porous_domain, "RT", 1)
Qp = FunctionSpace(porous_domain, "DG", 0)
U = VectorFunctionSpace(porous_domain, "CG", 2)

# # stokes
Vf = VectorFunctionSpace(stokes_domain, "CG", 2)
Qf = FunctionSpace(stokes_domain, "CG", 1)

# # lagrange multiplier
X = FunctionSpace(interface, "DG", 1)

W = [Vp, Qp, U, Vf, Qf, X]


## this is where the troubles start
up, pp, dp, uf, pf, lbd = map(TrialFunction, W)
vp, wp, ep, vf, wf, mu = map(TestFunction, W)

up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev = map(Function, W)
# up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev = ii_Function(W)



Tup, Tdp, Tuf = map(lambda x: Trace(x, interface),
                    [up, dp, uf]
)
        
Tvp, Tep, Tvf = map(lambda x: Trace(x, interface),
                    [vp, ep, vf]
)



dxGamma = Measure("dx", domain=interface)
n_Gamma_f = OuterNormal(interface, [0.5, 0.5, 0.5])
n_Gamma_p = -n_Gamma_f# L_mult = inner(Trace(dp, interface), n_Gamma_p) * mu * dxGamma
AL_mult = ii_assemble(inner(Trace(dp, interface), n_Gamma_p) * mu * dxGamma)

b = AL_mult*dp_prev.vector()

assert b.size() == X.dim()

import numpy as np

L_mult = inner(Trace(dp_prev, interface), n_Gamma_p) * mu * dxGamma
AL_mult = ii_assemble(inner(Trace(dp, interface), n_Gamma_p) * mu * dxGamma)

for i in range(3):
    dp_prev.vector().set_local(np.random.rand(dp_prev.vector().local_size()))
    dp_prev.vector().apply('insert')

    x = ii_assemble(L_mult)

    y = AL_mult*dp_prev.vector()

    assert (ii_convert(x) - ii_convert(y)).norm('linf') < DOLFIN_EPS

