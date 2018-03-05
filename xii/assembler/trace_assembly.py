from ufl.corealg.traversal import traverse_unique_terminals
from ufl.algorithms.transformer import ufl2uflcopy
from block import block_transpose
from ufl.form import Form
import dolfin as df
import operator

from xii.assembler.trace_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.trace_matrix import trace_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class TraceFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('trace_', )

    def select_integrals(self, form):
        return trace_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals if topological_dim(t) > tdim)
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes)
        assert terminal.trace_['mesh'].id() == reduced_mesh.id()
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        rtype = terminal.trace_['type']
        normal = terminal.trace_['normal'] if rtype else None

        return {'restriction': rtype, 'normal': normal}
    
    def reduced_space(self, V, reduced_mesh):
        '''Construct a reduced space for V on the mesh'''
        return trace_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return trace_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=TraceFormAssembler()):
    return assembler.assemble(form, arity)
