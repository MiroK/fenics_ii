from ufl_legacy.corealg.traversal import traverse_unique_terminals
from ufl_legacy.algorithms.transformer import ufl2uflcopy
from ufl_legacy.algorithms import estimate_total_polynomial_degree
from ufl_legacy import UFLException
from block import block_transpose
from ufl_legacy.form import Form
import dolfin as df
import operator

from xii.assembler.trace_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.trace_matrix import trace_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler
from xii.assembler.eval_dlt_trace_matrix import eval_dlt_trace_mat


class TraceFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('trace_', )

    def select_integrals(self, form):
        return trace_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals if topological_dim(t) > tdim)
    
    def is_compatible(self, terminal, reduced_mesh):
        return (all(hasattr(terminal, attr) for attr in self.attributes)
                and terminal.trace_['mesh'].id() == reduced_mesh.id())

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        rtype = terminal.trace_['type']
        normal = terminal.trace_['normal'] if rtype else None
        tag_data = terminal.trace_['tag']
        expr = terminal.trace_['expr']

        return {'restriction': rtype, 'normal': normal, 'tag_data': tag_data, 'expr': expr}
    
    def reduced_space(self, V, reduced_mesh, data=None):
        '''Construct a reduced space for V on the mesh'''
        if data is None:
            return trace_space(V, reduced_mesh)

        expr = data['expr']
        if expr is None:
            given_degree = None
        else:
            try:
                given_degree = estimate_total_polynomial_degree(expr)
            except UFLException:
                v, = filter(is_argument, traverse_unique_terminals(expr))
                given_degree = estimate_total_polynomial_degree(v)
        return trace_space(V, reduced_mesh, given_degree)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        if data['expr'] is None:
            return trace_mat(V, TV, reduced_mesh, data)

        return eval_dlt_trace_mat(V, TV, reduced_mesh, data)        

# Expose
    
def assemble_form(form, arity, assembler=TraceFormAssembler()):
    return assembler.assemble(form, arity)
