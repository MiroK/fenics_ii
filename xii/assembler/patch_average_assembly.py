from xii.assembler.patch_average_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.patch_average_matrix import patch_avg_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class PatchAverageFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('patch_average_', )

    def select_integrals(self, form):
        return patch_average_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        suspects = set(t for t in terminals if any(hasattr(t, attr) for attr in self.attributes))
        return suspects
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return {'vertex_f': terminal.patch_average_['vertex_f'],
                'patch_f': terminal.patch_average_['patch_f']}

    def reduced_space(self, V, reduced_mesh, data):
        '''Construct a reduced space for V on the mesh'''
        return patch_average_space(V, reduced_mesh, data)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return patch_avg_mat(V, TV, reduced_mesh, data)

# Expose
def assemble_form(form, arity, assembler=PatchAverageFormAssembler()):
    return assembler.assemble(form, arity)
