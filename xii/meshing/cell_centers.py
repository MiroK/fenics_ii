import dolfin as df
import numpy as np

# Let's construct the expression for computing circumcenter
from dolfin import CompiledExpression, compile_cpp_code

code_1_2 = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class Circumcenter_1_2 : public dolfin::Expression
{
public:

  Circumcenter_1_2() : dolfin::Expression(2) {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const dolfin::Cell cell(*mesh, cell_index);

    std::vector<double> coordinates;
    cell.get_vertex_coordinates(coordinates);

    Eigen::Vector2d v0(coordinates[0], coordinates[1]);
    Eigen::Vector2d v1(coordinates[2], coordinates[3]);
    Eigen::Vector2d y = 0.5*(v0 + v1);
    // Just midpoint
    values[0] = y[0];
    values[1] = y[1];
  }

  std::shared_ptr<dolfin::Mesh> mesh;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Circumcenter_1_2, std::shared_ptr<Circumcenter_1_2>, dolfin::Expression>
    (m, "Circumcenter_1_2")
    .def(py::init<>())
    .def_readwrite("mesh", &Circumcenter_1_2::mesh);
}
"""

code_1_3 = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class Circumcenter_1_3 : public dolfin::Expression
{
public:

  Circumcenter_1_3() : dolfin::Expression(3) {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const dolfin::Cell cell(*mesh, cell_index);

    std::vector<double> coordinates;
    cell.get_vertex_coordinates(coordinates);

    Eigen::Vector2d v0(coordinates[0], coordinates[1], coordinates[2]);
    Eigen::Vector2d v1(coordinates[3], coordinates[4], coordinates[5]);
    Eigen::Vector2d y = 0.5*(v0 + v1);
    // Just midpoint
    values[0] = y[0];
    values[1] = y[1];
  }

  std::shared_ptr<dolfin::Mesh> mesh;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Circumcenter_1_3, std::shared_ptr<Circumcenter_1_3>, dolfin::Expression>
    (m, "Circumcenter_1_3")
    .def(py::init<>())
    .def_readwrite("mesh", &Circumcenter_1_3::mesh);
}
"""

# Given simplex with vertices v0, v1, .... we consider a coordinate system 
# with origin in v0 and define spanning vectors u1 = v1-v0, u0 = v2-v0 as a coordinate
# system. In this one we seek a point c = v0 + Sum_k y_k *u_k such that
# distance(c, vi) is the same for all i. This leads to a linear system ...
code_2_2 = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class Circumcenter_2_2 : public dolfin::Expression
{
public:

  Circumcenter_2_2() : dolfin::Expression(2) {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const dolfin::Cell cell(*mesh, cell_index);

    std::vector<double> coordinates;
    cell.get_vertex_coordinates(coordinates);

    Eigen::Vector2d v0(coordinates[0], coordinates[1]);
    Eigen::Vector2d v1(coordinates[2], coordinates[3]);
    Eigen::Vector2d v2(coordinates[4], coordinates[5]);

    Eigen::Vector2d u = v1 - v0;
    Eigen::Vector2d v = v2 - v0;

    Eigen::Vector2d b(u.dot(u), v.dot(v));

    Eigen::Matrix2d A;
    A << u.dot(u), u.dot(v), v.dot(u), v.dot(v);

    Eigen::Vector2d yloc = A.fullPivLu().solve(0.5*b);
    Eigen::Vector2d y = v0 + yloc[0]*u + yloc[1]*v;

    values[0] = y[0];
    values[1] = y[1];

  }

  std::shared_ptr<dolfin::Mesh> mesh;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Circumcenter_2_2, std::shared_ptr<Circumcenter_2_2>, dolfin::Expression>
    (m, "Circumcenter_2_2")
    .def(py::init<>())
    .def_readwrite("mesh", &Circumcenter_2_2::mesh);
}
"""

code_2_3 = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class Circumcenter_2_3 : public dolfin::Expression
{
public:

  Circumcenter_2_3() : dolfin::Expression(3) {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const dolfin::Cell cell(*mesh, cell_index);

    std::vector<double> coordinates;
    cell.get_vertex_coordinates(coordinates);

    Eigen::Vector3d v0(coordinates[0], coordinates[1], coordinates[2]);
    Eigen::Vector3d v1(coordinates[3], coordinates[4], coordinates[5]);
    Eigen::Vector3d v2(coordinates[6], coordinates[7], coordinates[8]);

    Eigen::Vector3d u = v1 - v0;
    Eigen::Vector3d v = v2 - v0;

    Eigen::Vector2d b(u.dot(u), v.dot(v));

    Eigen::Matrix2d A;
    A << u.dot(u), u.dot(v), v.dot(u), v.dot(v);

    Eigen::Vector2d yloc = A.fullPivLu().solve(0.5*b);
    Eigen::Vector3d y = v0 + yloc[0]*u + yloc[1]*v;

    values[0] = y[0];
    values[1] = y[1];
    values[2] = y[2];
  }

  std::shared_ptr<dolfin::Mesh> mesh;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Circumcenter_2_3, std::shared_ptr<Circumcenter_2_3>, dolfin::Expression>
    (m, "Circumcenter_2_3")
    .def(py::init<>())
    .def_readwrite("mesh", &Circumcenter_2_3::mesh);
}
"""

# ------

code_3_3 = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>

class Circumcenter_3_3 : public dolfin::Expression
{
public:

  Circumcenter_3_3() : dolfin::Expression(3) {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const dolfin::Cell cell(*mesh, cell_index);

    std::vector<double> coordinates;
    cell.get_vertex_coordinates(coordinates);

    Eigen::Vector3d v0(coordinates[0], coordinates[1], coordinates[2]);
    Eigen::Vector3d v1(coordinates[3], coordinates[4], coordinates[5]);
    Eigen::Vector3d v2(coordinates[6], coordinates[7], coordinates[8]);
    Eigen::Vector3d v3(coordinates[9], coordinates[10], coordinates[11]);

    Eigen::Vector3d u = v1 - v0;
    Eigen::Vector3d v = v2 - v0;
    Eigen::Vector3d w = v3 - v0;

    Eigen::Vector3d b(u.dot(u), v.dot(v), w.dot(w));

    Eigen::Matrix3d A;
    A << u.dot(u), u.dot(v), u.dot(w), v.dot(u), v.dot(v), v.dot(w), w.dot(u), w.dot(v), w.dot(w); 

    Eigen::Vector3d yloc = A.fullPivLu().solve(0.5*b);
    Eigen::Vector3d y = v0 + yloc[0]*u + yloc[1]*v + yloc[2]*w;

    values[0] = y[0];
    values[1] = y[1];
    values[2] = y[2];
  }

  std::shared_ptr<dolfin::Mesh> mesh;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Circumcenter_3_3, std::shared_ptr<Circumcenter_3_3>, dolfin::Expression>
    (m, "Circumcenter_3_3")
    .def(py::init<>())
    .def_readwrite("mesh", &Circumcenter_3_3::mesh);
}
"""
codes = {(1, 2): code_1_2, (1, 3): code_1_3,
         (2, 3): code_2_3, (2, 2): code_2_2,
         (3, 3): code_3_3}


def CellCircumcenter(mesh, codes=codes):
    '''Circumcenter of cells in mesh'''
    assert mesh.ufl_cell().cellname() in ('interval', 'triangle', 'tetrahedron')
    
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()
    
    f = CompiledExpression(getattr(compile_cpp_code(codes[(tdim, gdim)]), f'Circumcenter_{tdim}_{gdim}')(),
                           degree=0, mesh=mesh)

    V = df.VectorFunctionSpace(mesh, 'DG', 0)
    return df.interpolate(f, V)


def _CenterVector(mesh, Center):
    '''DLT vector pointing on each facet from one Center to the other'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of circumcenters. For exterior it is facet centor to circumcenter
    # For facet centers we use DLT projection
    if mesh.topology().dim() > 1:
        L = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    else:
        L = df.VectorFunctionSpace(mesh, 'DG', 0)
        
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)

    facet_centers = FacetCentroid(mesh)
    cell_centers = Center(mesh)
    
    cc = df.Function(L)
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    df.assemble((1/fK('+'))*df.inner(cell_centers('+')-cell_centers('-'), l('+'))*df.dS +
                (1/fK)*df.inner(cell_centers-facet_centers, l)*df.ds,
                tensor=cc.vector())

    return cc



def _CenterDistance(mesh, Center):
    '''Magnitude of Centervector as a DLT function'''
    if mesh.topology().dim() > 1:
        L = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    else:
        L = df.FunctionSpace(mesh, 'DG', 0)
        
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)
    
    cc = Center(mesh)
    distance = df.Function(L)
    # We use P0 projection
    df.assemble(1/fK('+')*df.inner(df.sqrt(df.dot(cc('+'), cc('+'))), l('+'))*df.dS
                + 1/fK*df.inner(df.sqrt(df.dot(cc, cc)), l)*df.ds,
                tensor=distance.vector())
        
    return distance    


def _CenterDistance2(mesh, Center):
    '''Sum Magnitude of CenterVectors as a DLT function'''
    if mesh.topology().dim() > 1:
        L = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    else:
        L = df.FunctionSpace(mesh, 'DG', 0)
        
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)
    
    cc = Center(mesh)
    cc0, cc1 = df.split(cc)

    fc = FacetCentroid(mesh)
    
    distance = df.Function(L)
    # We use P0 projection
    df.assemble(1/fK('+')*df.inner(df.sqrt(df.dot(cc0('+')-fc('+'), cc0('+')-fc('+'))), l('+'))*df.dS +
                1/fK('+')*df.inner(df.sqrt(df.dot(cc1('-')-fc('-'), cc1('-')-fc('-'))), l('+'))*df.dS
                # NOTE: on the boundary the cc1 is 0
                + df.Constant(0.5)*(1/fK*df.inner(df.sqrt(df.dot(cc0-fc, cc0-fc)), l)*df.ds+
                                    1/fK*df.inner(df.sqrt(df.dot(cc1-fc, cc1-fc)), l)*df.ds),
                tensor=distance.vector())

    return distance

# -----------------------------------------------------------------------------

def _CenterVectors(mesh, Center):
    '''DLT ...'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of circumcenters. For exterior it is facet centor to circumcenter
    # For facet centers we use DLT projection
    cell = mesh.ufl_cell()
    if mesh.topology().dim() > 1:
        elm = df.VectorElement('HDiv Trace', cell, 0)
    else:
        elm = df.VectorElement('Lagrange', cell, 1)
    LL = df.FunctionSpace(mesh, df.MixedElement([elm, elm]))
    L = LL.sub(0).collapse()
                          
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)

    cell_centers = Center(mesh)

    plus = df.Function(L)
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    df.assemble((1/fK('+'))*df.inner(cell_centers('+'), l('+'))*df.dS +
                (1/fK)*df.inner(cell_centers, l)*df.ds,
                tensor=plus.vector())
    assert plus.vector().norm('linf') > 0
    
    minus = df.Function(L)    
    df.assemble((1/fK('-'))*df.inner(cell_centers('-'), l('-'))*df.dS +
                (1/fK)*df.inner(cell_centers, l)*df.ds,
                tensor=minus.vector())
    assert minus.vector().norm('linf') > 0
    
    cc = df.Function(LL)
    df.assign(cc, [plus, minus])

    assert cc.vector().norm('linf') > 0    

    return cc

# --------------------------------------------------------------------

def CircumDistance(mesh):
    '''Magnitude of Circumvector as a DLT function'''
    return _CenterDistance(mesh, Center=CircumVector)

                           
def CircumVector(mesh):
    '''DLT vector pointing on each facet from one circumcenter to the other'''
    return _CenterVector(mesh, Center=CellCircumcenter)

# --------------------------------------------------------------------

def CircumDistance2(mesh):
    '''Magnitude of Circumvector as a DLT function'''
    return _CenterDistance2(mesh, Center=CircumVector2)

                           
def CircumVector2(mesh):
    '''DLT vector pointing on each facet from one circumcenter to the other'''
    return _CenterVectors(mesh, Center=CellCircumcenter)

# --------------------------------------------------------------------

def CellCentroid(mesh):
    '''[DG0]^d function that evals on cell to its center of mass'''
    V = df.VectorFunctionSpace(mesh, 'DG', 0)
    v = df.TestFunction(V)

    hK = df.CellVolume(mesh)
    x = df.SpatialCoordinate(mesh)

    c = df.Function(V)
    df.assemble((1/hK)*df.inner(x, v)*df.dx, tensor=c.vector())

    return c


def CentroidDistance(mesh):
    '''Magnitude of CentroidVector as a DLT function'''
    return _CenterDistance(mesh, Center=CentroidVector)

                           
def CentroidVector(mesh):
    '''DLT vector pointing on each facet from one centroid to the other'''
    return _CenterVector(mesh, Center=CellCentroid)


def CentroidDistance2(mesh):
    '''Magnitude of CentroidVector as a DLT function'''
    return _CenterDistance(mesh, Center=CentroidVector2)

                           
def CentroidVector2(mesh):
    '''DLT vector pointing on each facet from one centroid to the other'''
    return _CenterVectors(mesh, Center=CellCentroid)

# --------------------------------------------------------------------

def FacetCentroid(mesh):
    '''[DLT]^d function'''
    xs = df.SpatialCoordinate(mesh)

    if mesh.topology().dim() == 1:
        V = df.FunctionSpace(mesh, 'Lagrange', 1)
    else:
        V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    v = df.TestFunction(V)
    hF = df.FacetArea(mesh)

    xi_foos = []
    for xi in xs:
        form = (1/df.avg(hF))*df.inner(xi, df.avg(v))*df.dS + (1/hF)*df.inner(xi, v)*df.ds
        xi = df.assemble(form)

        xi_foo = df.Function(V)
        xi_foo.vector()[:] = xi
        xi_foos.append(xi_foo)

    if mesh.topology().dim() == 1:
        V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    else:
        V = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        
    x = df.Function(V)
    for i, xi in enumerate(xi_foos):
        df.assign(x.sub(i), xi)

    return x


def is_delaunay_interior(mesh, _tol=1E-10):
    '''We are always parallel with facet normal vector'''
    n = df.FacetNormal(mesh)
    nu = CircumVector(mesh)   # Is this what we want?

    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    q = df.TestFunction(V)

    foo = df.assemble(df.inner(q('+'), df.sqrt(df.dot(nu('+'), nu('+'))))*df.dS
                      +df.inner(q, df.sqrt(df.dot(nu, nu)))*df.ds)
    print(np.where(np.abs(foo.get_local()) < 1E-10), V.dim())
    
    hF = df.FacetArea(mesh)

    nu = nu/df.sqrt(df.dot(nu, nu))
    # Compute the dot product between vector connecting points and the facet normal
    L = (1/hF('+'))*df.inner(df.dot(n('+'), nu('+')), q('+'))*df.dS
    # We are interested in this only on the interior facets
    Lmask = (1/hF('+'))*df.inner(df.dot(n('+'), n('+')), q('+'))*df.dS
    
    f = df.Function(V)
    df.assemble(L, f.vector())

    g = f.copy()
    df.assemble(Lmask, g.vector())
    mask = np.abs(g.vector().get_local() > _tol)

    assert np.linalg.norm(f.vector().get_local()[~mask], 2) < _tol
    # Ideally these guys will now be close to 1 in abslolute value
    coefs = f.vector().get_local()[mask]
    coefs = np.abs(coefs)

    return np.linalg.norm(np.abs(coefs-1), np.inf)
    
    
def is_delaunay_exterior(mesh, _tol=1E-10):
    '''We are always parallel with facet normal vector'''
    n = df.FacetNormal(mesh)
    nu = CircumVector(mesh)   # Is this what we want?
    nu = nu/df.sqrt(df.dot(nu, nu))

    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    q = df.TestFunction(V)

    hF = df.FacetArea(mesh)
    # Compute the dot product between vector connecting points and the facet normal
    L = (1/hF)*df.inner(df.dot(n, nu), q)*df.ds
    # We are interested in this only on the interior facets
    Lmask = (1/hF)*df.inner(df.dot(n, n), q)*df.ds
    
    f = df.Function(V)
    df.assemble(L, f.vector())

    g = f.copy()
    df.assemble(Lmask, g.vector())
    mask = np.abs(g.vector().get_local()) > _tol

    assert np.linalg.norm(f.vector().get_local()[~mask], 2) < _tol
    # Ideally these guys will now be close to 1 in abslolute value
    coefs = f.vector().get_local()[mask]
    coefs = np.abs(coefs)

    return np.linalg.norm(np.abs(coefs-1), np.inf)


def is_delaunay(mesh, _tol=1E-10):
    '''Look at the property separately on interior and exterior'''
    return (is_delaunay_interior(mesh, _tol),
            is_delaunay_exterior(mesh, _tol))

# --------------------------------------------------------------------

if __name__ == '__main__':
    mesh = df.UnitSquareMesh(1, 1)
    
    c = FacetCentroid(mesh)
    print(c.vector().get_local().reshape((-1, 2)))
