from xii.assembler.average_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.average_matrix import avg_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class AverageFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('average_', )

    def select_integrals(self, form):
        return average_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals if (topological_dim(t) - 2) == tdim)
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        assert terminal.average_['mesh'].id() == reduced_mesh.id()
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return {'bdry_curve': terminal.average_['bdry_curve']}

    def reduced_space(self, V, reduced_mesh):
        '''Construct a reduced space for V on the mesh'''
        return average_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return avg_mat(V, TV, reduced_mesh, data, which='surface')

# Expose
    
def assemble_form(form, arity, assembler=AverageFormAssembler()):
    return assembler.assemble(form, arity)
