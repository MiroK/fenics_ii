import xii.assembler.trace_assembly
import xii.assembler.average_assembly
import xii.assembler.restriction_assembly
from xii.linalg.matrix_utils import is_number
from xii.assembler.ufl_utils import form_arity

from block import block_vec, block_mat
from ufl.form import Form
import dolfin as df
import numpy as np


def assemble(form):
    '''Assemble multidimensional form'''
    # In the base case we want to fall trough the custom assemblers
    # for trace/average/restriction problems until something that 
    # dolfin can handle (hopefully)
    modules = (xii.assembler.trace_assembly,        # Codimension 1
               xii.assembler.average_assembly,      # Codimension 2
               xii.assembler.restriction_assembly)  # 0
    if isinstance(form, Form):
        arity = form_arity(form)
        # Try with our reduced assemblers
        for module in modules:
            tensor = module.assemble_form(form, arity)
            if tensor is not None:
                return tensor
        # Fallback
        return df.assemble(form)

    # We might get number
    if is_number(form): return form

    # Recurse
    if isinstance(form, list): form = np.array(form, dtype='object')

    blocks = np.array(map(assemble, form.flatten())).reshape(form.shape)
    
    return (block_vec if blocks.ndim == 1 else block_mat)(blocks)
