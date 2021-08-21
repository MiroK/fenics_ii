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
    
    return f


def _CenterVector(mesh, Center):
    '''DLT vector pointing on each facet from one Center to the other'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of circumcenters. For exterior it is facet centor to circumcenter
    V = df.VectorFunctionSpace(mesh, 'DG', 0)
    cell_centers = df.interpolate(Center(mesh), V)

    # For facet centers we use DLT projection
    L = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)

    x = df.SpatialCoordinate(mesh)
    facet_centers = df.Function(L)
    df.assemble((1/fK)*df.inner(x, l)*df.ds, tensor=facet_centers.vector())
        
    cc = df.Function(L)
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    df.assemble((1/fK('+'))*df.inner(cell_centers('+')-cell_centers('-'), l('+'))*df.dS +
                (1/fK)*df.inner(cell_centers-facet_centers, l)*df.ds,
                tensor=cc.vector())

    return cc


def _CenterDistance(mesh, Center):
    '''Magnitude of Centervector as a DLT function'''
    L = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)
    
    cc = Center(mesh)
    distance = df.Function(L)
    # We use P0 projection
    df.assemble(1/fK('+')*df.inner(df.sqrt(df.dot(cc('+'), cc('+'))), l('+'))*df.dS
                + 1/fK*df.inner(df.sqrt(df.dot(cc, cc)), l)*df.ds,
                tensor=distance.vector())
        
    return distance    


def CircumDistance(mesh):
    '''Magnitude of Circumvector as a DLT function'''
    return _CenterDistance(mesh, Center=CircumVector)

                           
def CircumVector(mesh):
    '''DLT vector pointing on each facet from one circumcenter to the other'''
    return _CenterVector(mesh, Center=CellCircumcenter)

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
