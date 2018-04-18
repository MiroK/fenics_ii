from block import block_mat, block_vec, block_bc
from dolfin import *
from xii import *

import mshr

dt, alpha, alpha_BJS, s0, mu_f, mu_p, lbd_f, lbd_p, K, Cp = [1] * 10

N = 10
EPS = 1E-3
R = 0.25
# box_domain = mshr.Box(dolfin.Point(0, 0, 0), dolfin.Point(1, 1, 1))
# _mesh = mshr.generate_mesh(box_domain, N)

_mesh = dolfin.UnitSquareMesh(10, 10)

subdomains = CellFunction('size_t', _mesh, 0)
dolfin.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(subdomains, 1)

surfaces = FacetFunction('size_t', _mesh, 0)
dolfin.CompiledSubDomain('near(x[0], 0.5)').mark(surfaces, 1)

stokes_domain = EmbeddedMesh(subdomains, 1)
porous_domain = EmbeddedMesh(subdomains, 0)
interface = EmbeddedMesh(surfaces, 1)

# function spaces

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


# this is where the troubles start
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
n_Gamma_f = OuterNormal(interface, [0.5, 0.5])
n_Gamma_p = -n_Gamma_f

############ a bunch of bilinear forms start here ############

mpp = Constant(s0 / dt) * pp * wp * dx
adp = Constant(mu_p / K) * inner(up, vp) * dx
aep = (
    Constant(mu_p / dt) * inner(sym(grad(dp)), sym(grad(ep))) * dx
    + Constant(lbd_p) * inner(div(dp), div(ep)) * dx
)
af = Constant(2 * mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dx

bpvp = - inner(div(vp), pp) * dx
bpvpt = - inner(div(up), wp) * dx
bpep = - Constant(alpha / dt) * inner(div(ep), pp) * dx
bpept = - Constant(alpha / dt) * inner(div(dp), wp) * dx
bf = - inner(div(vf), pf) * dx
bft = - inner(div(uf), wf) * dx

# matrices living on the interface
npvp, npep, nfvf = [lbd * dot(testfunc, n) * dxGamma
                    for (testfunc, n) in [(Tvp, n_Gamma_p), (Tep, n_Gamma_p), (Tvf, n_Gamma_f)]]
npvpt, npept, nfvft = [mu * dot(trialfunc, n) * dxGamma
                       for (trialfunc, n) in [(Tup, n_Gamma_p), (Tdp, n_Gamma_p), (Tuf, n_Gamma_f)]]

# sum_j ((a*tau_j), (b*tau_j))
svfuf, svfdp, sepuf, sepdp = [

    inner(testfunc, trialfunc) * Constant(1 / dt) * dxGamma
    - inner(testfunc, n_Gamma_f) * inner(trialfunc,
                                         n_Gamma_f) * Constant(1 / dt) * dxGamma
    for (testfunc, trialfunc) in [
        (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
    ]
]

############ no more bilinear forms ############


## this is the matrix
a = [
    [adp, bpvp, 0, 0, 0, npvp],
    [bpvpt, mpp, bpept, 0, 0, 0],
    [0, bpep, aep + sepdp, -sepuf, 0, Constant(1 / dt) * npep],
    [0, 0, -svfdp, af + Constant(dt) * svfuf, bf, nfvf],
    [0, 0, 0, bft, 0, 0],
    [npvpt, 0, Constant(1 / dt) * npept, nfvft, 0, 0],
]

AA = ii_convert(ii_assemble(a), "")
# NOTE: some conversion (typically with transpose) will leave
# the matrix in a wrong state from PETSc point of view (why?)
# From the experience the typical fix is setting the local-to-global
# map for the matrix. So we do it here just in case
set_lg_map(AA)

bcs = [
    [
        DirichletBC(
            W[0], Expression(("1", "1"), degree=2), "on_boundary"
        )
    ],
    [],
    [
        DirichletBC(
            W[2], Expression(("1", "1"), degree=2), "on_boundary"
        )
    ],
    [
        DirichletBC(
            W[3], Expression(("1", "1"), degree=2), "on_boundary"
        )
    ],
    [],
    [],
]

bbcs = block_bc(bcs, symmetric=False)
bbcs.apply(AA)
