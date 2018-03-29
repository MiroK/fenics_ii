from xii.assembler.point_trace_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.point_trace_matrix import point_trace_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class PointTraceFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('dirac_', )

    def select_integrals(self, form):
        return point_trace_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        '''Those who need to be reduced'''
        # Those that do not already 'live' on the reduced_mesh
        return set(t for t in terminals if hasattr(t, 'dirac_'))
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return terminal.dirac_

    def reduced_space(self, V, reduced_mesh):
        '''Construct a reduced space for V on the mesh'''
        return point_trace_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return point_trace_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=PointTraceFormAssembler()):
    return assembler.assemble(form, arity)
