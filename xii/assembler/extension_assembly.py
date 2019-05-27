from xii.assembler.extension_form import *
from xii.assembler.ufl_utils import *
from xii.assembler.extension_matrix import extension_mat
from xii.assembler.reduced_assembler import ReducedFormAssembler


class ExtensionFormAssembler(ReducedFormAssembler):
    def __init__(self):
        self.attributes = ('extension_', )

    def select_integrals(self, form):
        return extension_integrals(form)

    def restriction_filter(self, terminals, reduced_mesh):
        tdim = reduced_mesh.ufl_cell().topological_dimension()
        return set(t for t in terminals
                   if ((topological_dim(t)+1) == tdim) or (topological_dim(t) == tdim))
    
    def is_compatible(self, terminal, reduced_mesh):
        return (all(hasattr(terminal, attr) for attr in self.attributes)
                and terminal.extension_['mesh'].id() == reduced_mesh.id())

    def reduction_matrix_data(self, terminal):
        '''Dict of reduction data'''
        rtype = terminal.extension_['type']
        data = {'type': rtype}
        
        # Data for other extensions?
        if 'data' in terminal.extension_['type']:
            data['data'] = terminal.extension_['data']
        else:
            data['data'] = None
            
        return data 
    
    def reduced_space(self, V, extended_mesh):
        '''Construct a reduced space for V on the mesh'''
        return extension_space(V, extended_mesh)

    def reduction_matrix(self, V, TV, extended_mesh, data):
        '''Algebraic representation of the reduction'''
        return extension_mat(V, TV, extended_mesh, data)

# Expose
def assemble_form(form, arity, assembler=ExtensionFormAssembler()):
    return assembler.assemble(form, arity)
