import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from dolfin import cells
import numpy as np

# FIXME: vector?

def line_plot(f):
    '''Plots a function defined on a tdim1 mesh'''
    V = f.function_space()
    mesh = V.mesh()
    assert mesh.topology().dim() == 1, mesh.topology().dim()
    gdim = mesh.geometry().dim()
    assert gdim > 1
    
    rank = f.value_rank()
    # Scalar
    if rank == 0:
        return _line_plot_scalar(f)
    elif rank == 1:
        return _line_plot_vector(f)
    else:
        raise ValueError('No plots for tensors')


def _line_plot_scalar(f):
    '''Xd-1d plotting for scalar valued functions.'''
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    x = mesh.coordinates().reshape((-1, gdim))
    segments, z = [], []
    for cell in cells(mesh):
        line = x[cell.entities(0)]
        segments.append(line)

        mp = 0.5*(line[0]+line[1])
        value = f(mp)
        z.append(value)

    if gdim == 2:
        lc = LineCollection(np.array(segments), cmap=plt.get_cmap('hsv'))
    else:
        lc = Line3DCollection(np.array(segments), cmap=plt.get_cmap('hsv'))

    lc.set_array(np.array(z))
    lc.set_linewidth(3)

    fig = plt.figure()
    if gdim == 2:
        ax = fig.gca()
    else:
        ax = fig.gca(projection='3d')

    ax.add_collection(lc)
    fig.colorbar(lc)

    return fig


def _line_plot_vector(f):
    '''Xd-1d plotting for vector valued functions.'''
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    # First color segments by magnitude
    x = mesh.coordinates().reshape((-1, gdim))
    segments, mags, values, mps = [], [], [], []
    for cell in cells(mesh):
        line = x[cell.entities(0)]
        segments.append(line)

        mp = 0.5*(line[0]+line[1])
        value = f(mp)
        mag = np.linalg.norm(value)

        mags.append(mag)
        values.append(value)
        mps.append(mp)
    values = np.array(values)
    mps = np.array(mps)

    if gdim == 2:
        lc = LineCollection(np.array(segments), cmap=plt.get_cmap('hsv'))
    else:
        lc = Line3DCollection(np.array(segments), cmap=plt.get_cmap('hsv'))
    lc.set_array(np.array(mags))
    lc.set_linewidth(3)

    fig = plt.figure()
    if gdim == 2:
        ax = fig.gca()
        ax.quiver(mps[:, 0], mps[:, 1], values[:, 0], values[:, 1], pivot='tail')
    else:
        ax = fig.gca(projection='3d')
        ax.quiver3D(mps[:, 0], mps[:, 1], mps[:, 2], values[:, 0], values[:, 1],
                values[:, 2], pivot='tail')

    ax.add_collection(lc)
    fig.colorbar(lc)

    return fig

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    from embedded_mesh import EmbeddedMesh
    from dolfin import *

    if False:
        gamma = ['near((x[0]-0.25)*(x[0]-0.75), 0) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS)',
                 'near((x[1]-0.25)*(x[1]-0.75), 0) && (0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS)']
        gamma = map(lambda x: '('+x+')', gamma)
        gamma = ' || '.join(gamma)
        gamma = CompiledSubDomain(gamma)

        n = 20
        n *= 4
        omega_mesh = UnitSquareMesh(n, n)
        facet_f = FacetFunction('size_t', omega_mesh, 0)
        gamma.mark(facet_f, 1)

        gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

        # Q = FunctionSpace(gamma_mesh.mesh, 'DG', 1)
        # q = interpolate(Expression('std::sqrt(x[0]*x[0]+x[1]*x[1])', degree=2), Q)
        # fig = line_plot(q)
        # plt.show()

        Q = VectorFunctionSpace(gamma_mesh.mesh, 'DG', 1)
        q = interpolate(Expression(('x[0]', 'x[1]'), degree=1), Q)
        fig = line_plot(q)
        plt.show()

    if True:
        gamma = ['(near(x[0], x[1]) && near(x[1], x[2]))',
                 '(near(x[0], 1) && near(x[1], 1))',
                 '(near(x[0], x[1]) && near(x[2], 0))']
        gamma = ' || '.join(gamma)
        gamma = CompiledSubDomain(gamma)
        omega_mesh = UnitCubeMesh(10, 10, 10)

        facet_f = EdgeFunction('size_t', omega_mesh, 0)
        gamma.mark(facet_f, 1)

        gamma_mesh = EmbeddedMesh(omega_mesh, facet_f, 1)

        # Q = FunctionSpace(gamma_mesh.mesh, 'DG', 1)
        # q = interpolate(Expression('std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])', degree=2), Q)

        Q = VectorFunctionSpace(gamma_mesh.mesh, 'DG', 1)
        q = interpolate(Expression(('x[0]', 'x[1]', 'x[2]'), degree=1), Q)

        fig = line_plot(q)
        plt.show()
