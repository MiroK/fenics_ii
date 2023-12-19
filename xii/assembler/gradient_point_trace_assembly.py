from xii.assembler.gradient_point_trace_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.gradient_point_trace_matrix import gradient_point_trace_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class GradientPointTraceFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('grad_dirac_', )

    def select_integrals(self, form):
        return gradient_point_trace_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        '''Those who need to be reduced'''
        # Those that do not already 'live' on the reduced_mesh
        return set(t for t in terminals if hasattr(t, 'grad_dirac_'))
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return terminal.grad_dirac_

    def reduced_space(self, V, reduced_mesh, data=None):
        '''Construct a reduced space for V on the mesh'''
        return gradient_point_trace_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return gradient_point_trace_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=GradientPointTraceFormAssembler()):
    return assembler.assemble(form, arity)
