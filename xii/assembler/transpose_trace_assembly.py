from ufl_legacy.corealg.traversal import traverse_unique_terminals
from ufl_legacy.algorithms.transformer import ufl2uflcopy
from block import block_transpose
from ufl_legacy.form import Form
import dolfin as df
import operator

from xii.assembler.transpose_trace_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.transpose_trace_matrix import transpose_trace_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class TransposeTraceFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('transpose_trace_', )

    def select_integrals(self, form):
        return transpose_trace_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals if (topological_dim(t) +1) == tdim)
    
    def is_compatible(self, terminal, reduced_mesh):
          return all(hasattr(terminal, attr) for attr in self.attributes)
               
    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return {'trace_mesh': terminal.transpose_trace_['trace_mesh'],
                'full_mesh': terminal.transpose_trace_['full_mesh']}
    
    def reduced_space(self, V, reduced_mesh, data=None):
        '''Construct a reduced space for V on the mesh'''
        return transpose_trace_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return transpose_trace_mat(V, TV, reduced_mesh=data['trace_mesh'], tag_data=None)

# Expose
    
def assemble_form(form, arity, assembler=TransposeTraceFormAssembler()):
    return assembler.assemble(form, arity)
