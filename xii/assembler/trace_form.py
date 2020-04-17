from ufl.corealg.traversal import traverse_unique_terminals
from xii.assembler.ufl_utils import *
import dolfin as df
import ufl


def trace_cell(o):
    '''
    UFL cell corresponding to restriction of o[cell] to its facets, performing
    this restriction on o[function-like], or objects in o[function space]
    '''
    # Space
    if hasattr(o, 'ufl_cell'):
        return trace_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return trace_cell(o.ufl_element().cell())
    # Elm
    if hasattr(o, 'cell'):
        return trace_cell(o.cell())

    # Another cell
    cell_name = {'tetrahedron': 'triangle',
                 'triangle': 'interval'}[o.cellname()]

    return ufl.Cell(cell_name, o.geometric_dimension())


def trace_element(elm):
    '''
    Produce an intermerdiate element for computing with trace of 
    functions in FEM space over elm
    '''
    # Want exact match here; otherwise VectorElement is MixedElement and while
    # it works I don't find it pretty
    if type(elm) == df.MixedElement:
        return df.MixedElement(list(map(trace_element, elm.sub_elements())))
    
    # FIXME: Check out Witze Bonn's work on da Rham for trace spaces
    # in the meantime KISS
    family = elm.family()
    
    family_map = {'Lagrange': 'Lagrange'}
    # This seems like a reasonable fall back option
    family = family_map.get(family, 'Discontinuous Lagrange')

    degree = elm.degree()  # Preserve degree
    cell = trace_cell(elm)

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


def trace_space(V, mesh):
    '''
    Produce an intermerdiate function space for computing with trace of 
    functions in FEM space over elm
    '''    
    return df.FunctionSpace(mesh, trace_element(V.ufl_element()))


def Trace(v, mmesh, restriction='', normal=None, tag=None):
    '''
    Annotated function for being a restriction onto manifold of codimension
    one. By tag the trace can be localized on mmesh; then mmesh must have 
    a cell function with those markers
    '''
    # Prevent Trace(grad(u)). But it could be interesting to have this
    assert is_terminal(v)

    assert trace_cell(v) == mmesh.ufl_cell()
    # Not sure if it is really needed but will allow 5 types of traces
    assert restriction in ('',      # This makes sense for continuous foos
                           '+',     # For the remaining normal has to be
                           '-',     # present to get the orientation
                           'jump',  # right
                           'avg')
    # NOTE: for functions we make a new one which is annotated for restriction
    # but uses the same vector. This allows to make distinction e.g. in
    # differentiation. Crucially if Tu = Trace(u) the update to u of the form
    # u.vector()[:] = Vector(...) are reflected in Tu. WARNING: u.assign will
    # not work this way.
    # FIXME: should Trace(coeffcient) be a coefficient in the trace space right
    # away, what would be changes to the assembler etc?
    if isinstance(v, df.Coefficient):
        v =  df.Function(v.function_space(), v.vector())
    else:
        # Object copy?
        v = [df.TestFunction, df.TrialFunction][v.number()](v.function_space())

    if tag is not None:
        if isinstance(tag, int):
            tag = (tag, )
        tag = set(tag)
        
        assert hasattr(mmesh, 'marking_function')
        assert tag <= set(mmesh.marking_function.array())

        tag_data = (mmesh.marking_function, tag)
    else:
        # None means all
        tag_data = (df.MeshFunction('size_t', mmesh, mmesh.topology().dim(), 0),
                    set((0, )))
        
    v.trace_ = {'type': restriction, 'mesh': mmesh, 'normal': normal,
                'tag': tag_data}

    return v

# Consider now assembly of form, form is really a sum of integrals
# and here we want to assembly only the trace integrals. A trace integral
# is one where
#
# 0) the measure is the trace measure
#
# 1) all the Arguments are associated with a cell whose trace_cell is
#    a cell of the measure
#
# 2) all the Arguments are associated either with a cell that matches
#    the cell of the measure (do not need restriction) and those whose
#    trace_cell is that of the measure
#
# NOTE these are suspects. What I will check in the assembler is that
# each arg above was created by Trace
def is_trace_integrand(expr, tdim):
    '''Some of the arguments need restriction'''
    return any((topological_dim(arg)-1)  == tdim
               for arg in traverse_unique_terminals(expr))


def is_trace_integral(integral):
    '''Volume integral over an embedded cell'''
    return all((integral.integral_type() == 'cell',
                is_trace_integrand(integral.integrand(), topological_dim(integral))))


def trace_integrals(form):
    '''Extract trace integrals from the form'''
    return list(filter(is_trace_integral, form.integrals()))
