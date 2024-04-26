from xii.assembler.surface_mean_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.surface_mean_matrix import surface_mean_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class SurfaceMeanFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('surface_mean_', )

    def select_integrals(self, form):
        return surface_mean_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        '''Those who need to be reduced'''
        # Those that do not already 'live' on the reduced_mesh
        return set(t for t in terminals if hasattr(t, 'surface_mean_'))
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return terminal.surface_mean_

    def reduced_space(self, V, reduced_mesh, data=None):
        '''Construct a reduced space for V on the mesh'''
        return surface_mean_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return surface_mean_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=SurfaceMeanFormAssembler()):
    return assembler.assemble(form, arity)
