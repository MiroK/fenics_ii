from xii.assembler.ufl_utils import *
from xii.linalg.matrix_utils import is_number

from ufl.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl


def average_cell(o):
    '''
    UFL cell corresponding to restriction of o[cell] to its edges, performing
    this restriction on o[function-like], or objects in o[function space]
    '''
    # Space
    if hasattr(o, 'ufl_cell'):
        return average_cell(o.ufl_cell())
    # Foo like
    if hasattr(o, 'ufl_element'):
        return average_cell(o.ufl_element().cell())

    # Another cell
    cell_name = {'tetrahedron': 'interval'}[o.cellname()]
    
    return ufl.Cell(cell_name, o.geometric_dimension())


def average_space(V, mesh):
    '''Construct a space over mesh where surface averages of V should live'''
    # Sanity
    assert mesh.ufl_cell() == average_cell(V)

    elm = V.ufl_element()
    family = elm.family()
    degree = elm.degree()

    # Since tangent of a mesh cell/segment is uniquely defined only inside
    # the cell it is natural to represent everything in DG
    family = 'Discontinuous Lagrange'

    # There is an issue here where e.g. Hdiv are not scalars
    elmtype_map = {0: df.FiniteElement,
                   1: df.VectorElement,
                   2: df.TensorElement}
    # So let's check first for elements where scalar = FiniteElm
    # vector == VectorElm etc
    rank = len(elm.value_shape())
    if elmtype_map[rank] == type(elm):
        elm = type(elm)  # I.e. vector element stays verctor element
    else:
        elm = elmtype_map[rank]

    return df.FunctionSpace(mesh, elm(family, mesh.ufl_cell(), degree))


def Average(v, line_mesh, radius, quadrature_degree=8, surface='cylinder'):
    '''
    Annotated function for being surface average around line mesh. 
    For surface == 'cylinder' this means that 

        (Average(v))(x) = |C_R(x)|\int_{C_R(x)} v(y) dy for every point 

    x on the line_mesh. C_R(x) is a center of radius R(x) centered at x
    with normal vector determined by tangent of the line_mesh segment @ 
    x. For surface == 'sphere' we integrate over a surface of a ball 
    at x with radius. This integral is computed numerically with given quad.
    degree.

    If radius == 0, this reduction is understood as 3d-1d trace. In this 
    case the reduced function must be in some CG space!
    '''
    # Prevent Trace(grad(u)). But it could be interesting to have this
    assert is_terminal(v)
    assert average_cell(v) == line_mesh.ufl_cell()

    # Some sanity check for the radius
    if isinstance(radius, int) and radius == 0:
        v_family = v.ufl_element().family()
        # If we have en embedded mesh this mean that we want trace on en
        # edge and this make it well defined (only?) for CG
        if hasattr(line_mesh, 'parent_entity_map'):
            assert v_family == 'Lagrange', '3d1d trace undefined for %s' % v_family
        # Otherise the hope is that we will eval in cell interior which
        print '\tUsing 3d-1d trace!!!!'
        
        radius = None  # Signal to avg_mat
        
    if is_number(radius): assert radius > 0

    if isinstance(v, df.Coefficient):
        v =  df.Function(v.function_space(), v.vector())

    v.average_ = {'mesh': line_mesh, 'radius': radius, 'quad_degree': quadrature_degree,
                  'surface': surface}

    return v

# Consider now assembly of form, form is really a sum of integrals
# and here we want to assembly only the average integrals. An average integral
# is one where 
#
# 0) the measure is the average measure
#
# 1) all the Arguments are associated with a cell whose average_cell is
#    a cell of the measure
#
# 2) all the Arguments are associated either with a cell that matches
#    the cell of the measure (do not need restriction) and those whose
#    average_cell is that of the measure
#
# NOTE these are suspects. What I will check in the assembler is that
# each arg above was created by Average
def is_average_integrand(expr, tdim):
    '''Some of the arguments need restriction'''
    return any((topological_dim(arg) == tdim + 2) and isinstance(arg, Argument)
               for arg in traverse_unique_terminals(expr))


def is_average_integral(integral):
    '''Volume integral over an embedded line cell'''
    return all((integral.integral_type() == 'cell',  # 0
                (topological_dim(integral) + 2) == geometric_dim(integral),
                is_average_integrand(integral.integrand(), topological_dim(integral))))


def average_integrals(form):
    '''Extract trace integrals from the form'''
    return filter(is_average_integral, form.integrals())
