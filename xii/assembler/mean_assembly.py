from xii.assembler.mean_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.mean_matrix import mean_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class MeanFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('mean_', )

    def select_integrals(self, form):
        return mean_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        suspects = set(t for t in terminals if any(hasattr(t, attr) for attr in self.attributes))
        return suspects
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return terminal.mean_

    def reduced_space(self, V, reduced_mesh, data):
        '''Construct a reduced space for V on the mesh'''
        return mean_space(V, reduced_mesh, data)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return mean_mat(V, TV, reduced_mesh, data)

# Exposep
def assemble_form(form, arity, assembler=MeanFormAssembler()):
    return assembler.assemble(form, arity)
