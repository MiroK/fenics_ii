from xii.assembler.restriction_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.restriction_matrix import restriction_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class RestrictionFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('restriction_', )

    def select_integrals(self, form):
        return restriction_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        '''Those who need to be reduced'''
        # Those that do not already 'live' on the reduced_mesh
        have_different_mesh = set()
        for t in terminals:
            d = t.ufl_domain()
            if d is not None:  # Constants have None
                mesh = d.ufl_cargo()
                if mesh.id() != reduced_mesh.id():
                    have_different_mesh.add(t)

        return have_different_mesh
    
    def is_compatible(self, terminal, reduced_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes), self.attributes
        assert terminal.restriction_['mesh'].id() == reduced_mesh.id()
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data and optinal normal'''
        return {}

    def reduced_space(self, V, reduced_mesh):
        '''Construct a reduced space for V on the mesh'''
        return restriction_space(V, reduced_mesh)

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        return restriction_mat(V, TV, reduced_mesh, data)

# Expose
    
def assemble_form(form, arity, assembler=RestrictionFormAssembler()):
    return assembler.assemble(form, arity)
