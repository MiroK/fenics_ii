from xii.assembler.injection_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.injection_matrix import injection_matrix
from xii.assembler.reduced_assembler import ReducedFormAssembler


class InjectionFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('injection_', )

    def select_integrals(self, form):
        return injection_integrals(form)

    def restriction_filter(self, terminals, fine_mesh):
        '''Those are'''
        lives_on_coarse = set()
        for t in terminals:
            d = t.ufl_domain()
            if d is not None:  # Constants have None
                mesh = d.ufl_cargo()
                if mesh.id() != fine_mesh.id():
                    lives_on_coarse.add(t)
        return lives_on_coarse
    
    def is_compatible(self, terminal, fine_mesh):
        assert all(hasattr(terminal, attr) for attr in self.attributes), self.attributes
        assert terminal.injection_['mesh'].id() == fine_mesh.id()
        return True

    def reduction_matrix_data(self, terminal):
        '''Dict of injection data'''
        return {}

    def reduced_space(self, V, fine_mesh):
        '''Construct an injection space for V on the mesh'''
        return injection_space(V, fine_mesh)

    def reduction_matrix(self, Vc, Vf, fine_mesh, data):
        '''Algebraic representation of injection'''
        return injection_matrix(Vc, Vf)

# Expose
    
def assemble_form(form, arity, assembler=InjectionFormAssembler()):
    return assembler.assemble(form, arity)
