import xii.assembler.trace_assembly
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
    if isinstance(form, Form):
        arity = form_arity(form)
        if arity == 2:
            for module in (xii.assembler.trace_assembly, ): # average, restriction):
                A = module.assemble_bilinear_form(form)
                if A is not None:
                    return A
            
            return df.assemble(form)

        if arity == 1:
            for module in (trace_assembly, ): # average, restriction):
                b = module.assemble_linear_form(form)
                if b is not None:
                    return b
            
            return df.assemble(form)
         
    # We might get number
    if is_number(form): return form

    # Recurse
    if isinstance(form, list): form = np.array(form, dtype='object')

    blocks = np.array(map(assemble, form.flatten())).reshape(form.shape)

    return (block_vec if blocks.ndim == 1 else block_mat)(blocks)
