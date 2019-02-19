from ufl.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import ufl


def extension_cell(o):
    '''UFL cell corresponding to extension of o[cell] to facets in 3d'''
    # Space
    if hasattr(o, 'ufl_cell'):
        return extension_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return extension_cell(o.ufl_element().cell())
    # Elm
    if hasattr(o, 'cell'):
        return extension_cell(o.cell())

    assert o.geometric_dimension() == 3
    assert o.topological_dimension() == 1
    
    return ufl.Cell('triangle', o.geometric_dimension())


def extension_element(elm):
    '''
    Produce an intermerdiate element for computing with extension of 
    functions in FEM space over elm
    '''
    # Want exact match here; otherwise VectorElement is MixedElement and while
    # it works I don't find it pretty
    if type(elm) == df.MixedElement:
        return df.MixedElement(map(extension_element, elm.sub_elements()))
    

    family = elm.family()
    
    family_map = {'Lagrange': 'Lagrange'}
    # This seems like a reasonable fall back option
    family = family_map.get(family, 'Discontinuous Lagrange')

    degree = elm.degree()  # Preserve degree
    cell = extension_cell(elm)

    # How to construct:
    # There is an issue here where e.g. Hdiv are not scalars, their
    # element is FiniteElement but we want trace space from VectorElement
    elmtype_map = {0: df.FiniteElement,
                   1: df.VectorElement,
                   2: df.TensorElement}
    # So let's check first for elements where scalar = FiniteElm, vector == VectorElm 
    rank = len(elm.value_shape())
    if elmtype_map[rank] == type(elm):
        elm = type(elm)  # i.e. vector element stays vector element
    else:
        elm = elmtype_map[rank]

    return elm(family, cell, degree)


def extension_space(V, mesh):
    '''
    Produce an intermerdiate function space for computing with extension of 
    functions in FEM space over elm
    '''    
    return df.FunctionSpace(mesh, extension_element(V.ufl_element()))


def Extension(v, mesh, type):
    '''Annotated function for being an extension onto a 2d manifold.'''
    # Prevent Ext(grad(u)). But it could be interesting to have this
    assert is_terminal(v)

    assert extension_cell(v) == mesh.ufl_cell()

    if isinstance(v, df.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    v.extension_ = {'type': type, 'mesh': mesh}

    return v


def is_extension_integrand(expr, tdim):
    '''Some of the arguments need extension'''
    return any((topological_dim(arg)+1)  == tdim
               for arg in traverse_unique_terminals(expr))


def is_extension_integral(integral):
    '''Volume integral over an embedded cell'''
    return all((integral.integral_type() == 'cell',
                is_extension_integrand(integral.integrand(), topological_dim(integral))))


def extension_integrals(form):
    '''Extract trace integrals from the form'''
    return filter(is_extension_integral, form.integrals())
