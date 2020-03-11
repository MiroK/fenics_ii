from xii.linalg.matrix_utils import petsc_serial_matrix
from xii.assembler.fem_eval import DegreeOfFreedom, FEBasisFunction
from xii import *

from dolfin import PETScMatrix
from collections import defaultdict
from petsc4py import PETSc
import numpy as np


def injection_matrix(Vc, Vf):
    '''Injection mapping from Vc to Vf'''
    mesh_c = Vc.mesh()
    mesh_f = Vf.mesh()

    assert mesh_f.has_parent() and mesh_c.has_child()
    assert mesh_f.parent().id() == mesh_c.id()

    fine_to_coarse = mesh_f.data().array('parent_cell', mesh_f.topology().dim())
    
    # The idea is to evaluate Vf's degrees of freedom at basis functions of Vc
    fdmap = Vf.dofmap()
    Vf_dof = DegreeOfFreedom(Vf)

    cdmap = Vc.dofmap()
    Vc_basis_f = FEBasisFunction(Vc)

    # Column values
    visited_rows = [False]*Vf.dim()
    row_values = np.zeros(Vc_basis_f.elm.space_dimension(), dtype='double')

    with petsc_serial_matrix(Vf, Vc) as mat:
        
        for f_cell, c_cell in enumerate(fine_to_coarse):

            Vc_basis_f.cell = c_cell
            # These are the colums
            coarse_dofs = cdmap.cell_dofs(c_cell)

        
            Vf_dof.cell = f_cell
            
            fine_dofs = fdmap.cell_dofs(f_cell)

            for local, dof in enumerate(fine_dofs):
                if visited_rows[dof]:
                    continue
                else:
                    visited_rows[dof] = True
                            
                Vf_dof.dof = local
                # Evaluete coarse basis foos here
                for local_c, dof_c in enumerate(coarse_dofs):
                    Vc_basis_f.dof = local_c
                    row_values[local_c] = Vf_dof.eval(Vc_basis_f)
                # Insert
                mat.setValues([dof], coarse_dofs, row_values, PETSc.InsertMode.INSERT_VALUES)
    # Transpose
    return mat


def stokes(mesh_c):

    mesh_f = adapt(mesh_c)
    Vf = VectorFunctionSpace(mesh_f, 'CG', 1)
    # Pressure
    Qc = FunctionSpace(mesh_c, 'CG', 1)
    Qf = FunctionSpace(mesh_f, 'CG', 1)

    W = [Vf, Qc]

    Wf = [Vf, Qf]
    uf, pf = map(TrialFunction, Wf)
    vf, qf = map(TestFunction, Wf)

    a = [[0]*2 for i in range(2)]
    a[0][0] = inner(grad(uf), grad(vf))*dx
    a[0][1] = inner(pf, div(vf))*dx
    a[1][0] = inner(qf, div(uf))*dx

    # Auxliary coarse problem
    A = ii_assemble(a)

    J = PETScMatrix(injection_matrix(Qc, Qf))
    # We now modify the div block as if pressure was coarse
    A[0][1] = A[0][1]*J
    A[1][0] = J.T*A[1][0]

    V_bcs = DirichletBC(Vf, Constant((0, 0)), 'near(x[0], 0)')
    A, b = apply_bc(A, b=None, bcs=[V_bcs, []])

    # Preconditioner
    V_inner, _ = assemble_system(inner(grad(uf), grad(vf))*dx,
                                 inner(Constant((0, 0)), vf)*dx,
                                 bcs=V_bcs)

    pc, qc = TrialFunction(Qc), TestFunction(Qc)
    Q_inner = assemble(inner(pc, qc)*dx)

    B = block_diag_mat([V_inner, Q_inner])

    return A, B, W


# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from weak_bcs.la_solve import my_eigvalsh

    mesh_c = UnitSquareMesh(1, 1)
    mesh_f = adapt(mesh_c)
    
    Vc = FunctionSpace(mesh_c, 'CG', 1)
    Vf = FunctionSpace(mesh_f, 'CG', 1)

    Jmat = injection_matrix(Vc, Vf)
    J = PETScMatrix(Jmat)

    f = Expression('x[0]+x[1]', degree=1)
    x = interpolate(f, Vc).vector().get_local()
    y = J.array().dot(x)  # Inject coarse into fine
    y0 = interpolate(f, Vf).vector().get_local()
    # print y0, y

#    fc = sum(ff_i*c_i
#             c_i = dof_fi(phic_j)

    out = None
    for n in (2, 4, 8, 16, 32):
        mesh_c = UnitSquareMesh(n, n)
        A, B, W = stokes(mesh_c)

        A = ii_convert(A).array()
        B = ii_convert(B).array()

        lmin, lmax = np.sort(np.abs(my_eigvalsh(A, B)))[[0, -1]]
        print sum(Wi.dim() for Wi in W), '->', lmin, lmax, lmax/lmin

        Vf, _ = W
        mesh_f = Vf.mesh()
        values = mesh_f.data().array('parent_cell', mesh_f.topology().dim())
        cell_f = MeshFunction('size_t', mesh_f, mesh_f.topology().dim(), 0)
        cell_f.array()[:] = values

        if out is None:
            out = File('foo.pvd')
            out << cell_f
