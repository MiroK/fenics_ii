from ufl_legacy.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import ufl_legacy as ufl


def injection_cell(o):
    '''The same cell type but in a different mesh'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return injection_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return injection_cell(o.ufl_element().cell())

    # Another cell
    return o


def injection_space(V, mesh):
    '''Construct a space where restrictions of V to mesh should live'''
    # Sanity
    assert mesh.ufl_cell() == injection_cell(V)
    # All is the same
    return df.FunctionSpace(mesh, V.ufl_element())


def Injection(v, fmesh, not_nested_method='interpolate'):
    '''
    Annotated function for being a injection from space on coarse mesh 
    to space of fine mesh.
    '''
    assert is_terminal(v)
    assert injection_cell(v) == fmesh.ufl_cell()

    # Type check
    if hasattr(v, 'function_space'):
        cmesh = v.function_space().mesh()
        #
        has_data = hasattr(fmesh, 'parent_entity_map') and cmesh.id() in fmesh.parent_entity_map
        
        if not has_data:
            try:
                cmap = fmesh.data().array('parent_cell', fmesh.topology().dim())
                tdim = fmesh.topology().dim()
                # NOTE: there is no vertex map
                fmesh.parent_entity_map = {cmesh.id(): {tdim: dict(enumerate(cmap))}}
                has_data = True
                
            except RuntimeError:
                pass

        if has_data:
            assert cmesh.id() in fmesh.parent_entity_map
        else:
            assert not_nested_method in ('interpolate', 'project')

    if isinstance(v, ufl.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.injection_ = {'mesh': fmesh,
                    'not_nested_method': not_nested_method}
    
    return v


def is_injection(arg):
    '''Very crude check'''
    return hasattr(arg, 'injection_')


def is_injection_integrand(expr):
    '''Is it?'''
    return any((is_injection(arg) for arg in traverse_unique_terminals(expr)))


def is_injection_integral(integral):
    '''
    Cell integral over domain that is that over domain that has child
    '''
    fmesh = integral.ufl_domain().ufl_cargo()
    
    for arg in filter(is_injection, traverse_unique_terminals(integral.integrand())):
        if arg.injection_['mesh'].id() == fmesh.id():
            return True

    return False


def injection_integrals(form):
    '''Extract injection integrals from the form'''
    return list(filter(is_injection_integral, form.integrals()))
