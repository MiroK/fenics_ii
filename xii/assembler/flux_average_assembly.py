from xii.assembler.flux_average_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.flux_average_matrix import flux_avg_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class FluxAverageFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('flux_average_', )

    def select_integrals(self, form):
        return flux_average_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals if (topological_dim(t) - 2) == tdim)
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        assert terminal.flux_average_['mesh'].id() == reduced_mesh.id()
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return {'shape': terminal.flux_average_['shape']}

    def domain_space(self, terminal):
        '''Default to space where terminal is defined'''
        return terminal.flux_average_['domain']

    def reduced_space(self, V, reduced_mesh, data=None):
        '''Construct a reduced space for V on the mesh'''
        return flux_average_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return flux_avg_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=FluxAverageFormAssembler()):
    return assembler.assemble(form, arity)
