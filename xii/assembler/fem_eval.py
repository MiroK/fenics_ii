from __future__ import absolute_import
from __future__ import print_function
from dolfin import Cell, Expression
import dolfin as df
import numpy as np


def cell_orientation(is_1d_in_3d):
    '''A line mesh in 3d cannot be oriented - fall back to 0'''
    return lambda cell: cell.orientation() if not is_1d_in_3d else 0


class DegreeOfFreedom(object):
    '''Evaluator of dof of V on functions'''
    def __init__(self, V):
        self.elm = V.element()
        self.mesh = V.mesh()

        is_1d_in_3d = self.mesh.topology().dim() == 1 and self.mesh.geometry().dim() == 3
        self.orient_cell = cell_orientation(is_1d_in_3d)

        # Allocs
        self.__cell = Cell(self.mesh, 0)
        self.__cell_vertex_x = self.__cell.get_vertex_coordinates()
        self.__cell_orientation = self.orient_cell(self.__cell)
        self.__dof = 0

    @property
    def dof(self):
        return self.__dof

    @dof.setter
    def dof(self, value):
        assert value < self.elm.space_dimension()
        self.__dof = value
    
    @property
    def cell(self):
        return self.__cell

    @cell.setter
    def cell(self, value):
        cell_ = Cell(self.mesh, value)
        self.__cell_vertex_x[:] = cell_.get_vertex_coordinates()
        self.__cell_orientation = self.orient_cell(cell_)
        self.__cell = cell_

    def eval(self, f):
        if hasattr(self.elm, 'evaluate_dof'):
            return self.elm.evaluate_dof(self.dof,
                                         f.as_expression() if isinstance(f, FEBasisFunction) else f,
                                         self.__cell_vertex_x,
                                         self.__cell_orientation,
                                         self.__cell)
        else:
            return self.elm.evaluate_dofs(f.as_expression() if isinstance(f, FEBasisFunction) else f,
                                          self.__cell_vertex_x,
                                          self.__cell_orientation,
                                          self.__cell)[self.dof]

# NOTE: this is a very silly construction. Basically the problem is
# that Function cannot be properly overloaded beacsue SWIG does not 
# expose eval.
class FEBasisFunction(object):
    '''Evaluator of dof of V on functions'''
    def __init__(self, V):
        self.elm = V.element()
        self.mesh = V.mesh()

        shape = V.ufl_element().value_shape()
        degree = V.ufl_element().degree()

        # A fake instanc to talk to with the world
        adapter = type('MiroHack',
                       (Expression if not hasattr(df, 'UserExpression') else df.UserExpression, ),
                       {'value_shape': lambda self_, : shape,
                        'eval': lambda self_, values, x: self.eval(values, x)})
        self.__adapter = adapter(degree=degree)

        is_1d_in_3d = self.mesh.topology().dim() == 1 and self.mesh.geometry().dim() == 3
        self.orient_cell = cell_orientation(is_1d_in_3d)

        # Allocs
        self.__cell = Cell(self.mesh, 0)
        self.__cell_vertex_x = self.__cell.get_vertex_coordinates()
        self.__cell_orientation = self.orient_cell(self.__cell)
        self.__dof = 0
        self.__values = np.zeros(V.ufl_element().value_size())

    @property
    def dof(self):
        return self.__dof

    @dof.setter
    def dof(self, value):
        assert value < self.elm.space_dimension()
        self.__dof = value
    
    @property
    def cell(self):
        return self.__cell

    @cell.setter
    def cell(self, value):
        cell_ = Cell(self.mesh, value)
        self.__cell_vertex_x[:] = cell_.get_vertex_coordinates()
        self.__cell_orientation = self.orient_cell(cell_)
        self.__cell = cell_

    def eval(self, values, x):
        if hasattr(self.elm, 'evaluate_dof'):
            self.elm.evaluate_basis(self.dof,
                                    values[:], 
                                    x,
                                    self.__cell_vertex_x,
                                    self.__cell_orientation)
        else:
            values[:] = self.elm.evaluate_basis(self.dof,
                                                x,
                                                self.__cell_vertex_x,
                                                self.__cell_orientation)

    def __call__(self, x):
        self.eval(self.__values, x)
        return 1*self.__values

    def as_expression(self):
        return self.__adapter


# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *

    cell = 1
    local_dof = 0
    
    mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f_values = f.vector().get_local()

    f_values[V.dofmap().cell_dofs(cell)[local_dof]] = 1.0
    f.vector().set_local(f_values)

    g = FEBasisFunction(V)
    g.cell = cell
    g.dof = local_dof

    dofs = DegreeOfFreedom(V)
    dofs.cell = cell
    dofs.dof = local_dof

    print(dofs.eval(f), dofs.eval(g.as_expression()))

    x = Cell(mesh, cell).midpoint().array()[:2]
    print(f(x), g(x))
