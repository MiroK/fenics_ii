from ufl.corealg.traversal import traverse_unique_terminals
from ufl.algorithms.transformer import ufl2uflcopy
from block import block_transpose
from ufl.form import Form
import dolfin as df
import operator

from xii.assembler.trace_form import *
from xii.assembler.ufl_utils import *
import xii.assembler.xii_assembler
import xii.assembler.trace_matrix


def assemble_bilinear_form(form):
    '''
    Trace form that can be assembled must allow for the following representation
    of the assembled matrix: a*T' * assemble(reduced_form) b*T where 
    a, b in {0, 1} and (a or b).
    '''
    tintegrals = trace_integrals(form)
    # Signal to xii.assemble
    if not tintegrals: return None
    
    matrices = []

    for integral in form.integrals():
        # Delegate to friend
        if integral not in tintegrals:
            matrices.append(xii.assembler.xii_assembler.assemble(Form([integral])))
            continue
            
        # Get the tdim of integral
        tdim = integral.ufl_domain().topological_dimension()
        trace_mesh = integral.ufl_domain().ufl_cargo()

        integrand = integral.integrand()
        # Split arguments in those that need to be and those that are
        # already restricted.
        terminals = set(t for t in traverse_unique_terminals(integrand)
                        if is_trial_function(t) or is_test_function(t))

        trace_terminals = set(t for t in terminals if topological_dim(t) > tdim)
        # You said this is a trace ingral!
        assert len(trace_terminals) > 0
        # Now all the guys have to be annotated
        assert all(hasattr(t, 'trace_') for t in trace_terminals)
        # Let's finally make sure that their restriction is to the trace_mesh
        assert all(t.trace_['mesh'].id() == trace_mesh.id() for t in trace_terminals)

        # In accordance with the abstraction there should be only one R matrix
        # NOTE: combining several trace types is not allowed. The effect
        # can be achieved by spliting the form
        # FIXME: split the form if we detect this
        ttype = set(t.trace_['type'] for t in trace_terminals)
        assert len(ttype) == 1
        ttype = ttype.pop()

        # For ttype which requires normal the normal should then be the same
        if ttype:
            normal = set(t.trace_['normal'] for t in trace_terminals)
            assert len(normal) == 1
            normal = normal.pop()
        else:
            normal = None
                
        integrand = ufl2uflcopy(integrand)
        # The aT' * R * bT is not assembled as a matrix but a composition
        # which can be collaped later
        components = [1, 1, 1]
        for tterm in trace_terminals:
            # With sane inputs we can get the trace element and setup the
            # intermediate function space where the trace of trace terminal
            # lives
            V = tterm.function_space()
            TV = trace_space(V, trace_mesh)

            # Setup the matrix to from space of the trace_terminal to the
            # intermediate space. FIXME: normal and trace_mesh
            T = xii.assembler.trace_matrix.trace_mat(V, TV, restriction=ttype, normal=normal)

            if is_test_function(tterm):
                components[0] = block_transpose(T)
                replacement = df.TestFunction(TV)

            if is_trial_function(tterm):
                components[-1] = T
                replacement = df.TrialFunction(TV)
            
            # With substitution
            integrand = replace(integrand, tterm, replacement, attributes=('trace_', ))
        # Having performed substiturion we can get the redued form
        trace_form = Form([integral.reconstruct(integrand=integrand)])
        # Assemble the reduced form
        components[1] = xii.assembler.xii_assembler.assemble(trace_form)  # The R matrix
        # We combine the ingredients to get matrix representation of the
        # integral
        matrices.append(reduce(operator.mul, components))
    # The whole form is then the sum of integrals
    return reduce(operator.add, matrices)


def assemble_linear_form(form):
    '''
    Trace form that can be assembled must allow for the following representation
    of the assembled matrix: T' * assemble(reduced_form)
    '''
    tintegrals = trace_integrals(form)
    # Signal to xii.assemble
    if not tintegrals: return None
    
    vectors = []

    for integral in form.integrals():
        # Delegate to friend
        if integral not in tintegrals:
            matrices.append(xii.assembler.xii_assembler.assemble(Form([integral])))
            continue
            
        # Get the tdim of integral
        tdim = integral.ufl_domain().topological_dimension()
        trace_mesh = integral.ufl_domain().ufl_cargo()

        integrand = integral.integrand()
        # Split arguments in those that need to be and those that are
        # already restricted.
        terminals = set(t for t in traverse_unique_terminals(integrand)
                        if is_trial_function(t) or is_test_function(t))

        trace_terminals = set(t for t in terminals if topological_dim(t) > tdim)
        # You said this is a trace ingral!
        # More specifically
        assert trace_terminals == terminals
        assert len(trace_terminals) == 1

        tterm = trace_terminals.pop()
        # Now all the guys have to be annotated
        assert hasattr(tterm, 'trace_')
        # Let's finally make sure that their restriction is to the trace_mesh
        assert tterm.trace_['mesh'].id() == trace_mesh.id()

        ttype = tterm.trace_['type']
        # Extract normals if needed
        if ttype:
            normal = t.trace_['normal']
        else:
            normal = None

        integrand = ufl2uflcopy(integrand)
        # The terace space
        V = tterm.function_space()
        TV = trace_space(V, trace_mesh)
        # Setup the matrix to from space of the trace_terminal to the
        T = xii.assembler.trace_matrix.trace_mat(V, TV, restriction=ttype, normal=normal)

        replacement = df.TestFunction(TV)

        # With substitution
        integrand = replace(integrand, tterm, replacement, attributes=('trace_', ))

        trace_form = Form([integral.reconstruct(integrand=integrand)])
        # Assemble the reduced form
        b = xii.assembler.xii_assembler.assemble(trace_form)
        # The result it T' * b 
        Tb = df.Function(V).vector()
        T.transpmult(b, Tb)

        vectors.append(Tb)
    # The whole form is then the sum of integrals
    return reduce(operator.add, vectors)  # Sum
