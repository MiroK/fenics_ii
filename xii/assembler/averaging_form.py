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
