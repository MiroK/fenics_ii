from ufl.corealg.traversal import traverse_unique_terminals
from ufl.algorithms.transformer import ufl2uflcopy
from block import block_transpose
from ufl.form import Form
import dolfin as df
import operator

from xii.assembler.trace_form import *
from xii.assembler.ufl_utils import *
import xii.assembler.xii_assembly
from functools import reduce

class ReducedFormAssembler(object):
    '''
    We assemble the biliner form into a product of algebraic representation
    of the reduction operatrion and a assembled reduced bilinear form. 
    A linear form is reduction operation times vector.
    '''
    def select_integrals(self, form):
        '''Integrals to which the assembler can be applied'''
        raise NotImplementedError

    def restriction_filter(self, terminals, reduced_mesh):
        '''Given a set of terminals give me those that need restriction'''
        raise NotImplementedError
    
    def is_compatible(self, terminal, reduced_mesh):
        '''Sanity check'''
        raise NotImplementedError

    def reduction_matrix_data(self, terminal):
        '''
        Operator specific data for reduction of the terminal
        '''
        raise NotImplementedError
    
    def get_normal(self, terminal):
        '''Extract normal'''
        raise NotImplementedError

    def reduced_space(self, V, reduced_mesh):
        '''Construct a reduced space for V on the mesh'''
        raise NotImplementedError

    def reduction_matrix(self, V, TV, reduced_mesh, data):
        '''Algebraic representation of the reduction'''
        raise NotImplementedError

    # Common logic:
    def assemble(self, form, arity):
        '''Assemble a biliner(2), linear(1) form'''
        reduced_integrals = self.select_integrals(form)   #! Selector
        # Signal to xii.assemble
        if not reduced_integrals: return None
    
        components = []
        for integral in form.integrals():
            # Delegate to friend
            if integral not in reduced_integrals:
                components.append(xii.assembler.xii_assembly.assemble(Form([integral])))
                continue

            reduced_mesh = integral.ufl_domain().ufl_cargo()

            integrand = integral.integrand()
            # Split arguments in those that need to be and those that are
            # already restricted.
            terminals = set(traverse_unique_terminals(integrand))

            # FIXME: is it enough info (in general) to decide
            terminals_to_restrict = self.restriction_filter(terminals, reduced_mesh)
            # You said this is a trace ingral!
            assert terminals_to_restrict

            # Let's pick a guy for restriction
            terminal = terminals_to_restrict.pop()
            # We have some assumption on the candidate
            assert self.is_compatible(terminal, reduced_mesh)

            data = self.reduction_matrix_data(terminal)
                
            integrand = ufl2uflcopy(integrand)
            # With sane inputs we can get the reduced element and setup the
            # intermediate function space where the reduction of terminal
            # lives
            V = terminal.function_space()
            TV = self.reduced_space(V, reduced_mesh)  #! Space construc

            # Setup the matrix to from space of the trace_terminal to the
            # intermediate space. FIXME: normal and trace_mesh
            #! mat construct
            T = self.reduction_matrix(V, TV, reduced_mesh, data)

            # T
            if is_test_function(terminal):
                replacement = df.TestFunction(TV)
                # Passing the args to get the comparison a make substitution
                integrand = replace(integrand, terminal, replacement, attributes=self.attributes)
                trace_form = Form([integral.reconstruct(integrand=integrand)])

                if arity == 2:
                    # Make attempt on the substituted form
                    A = xii.assembler.xii_assembly.assemble(trace_form)
                    components.append(block_transpose(T)*A)
                else:
                    b = xii.assembler.xii_assembly.assemble(trace_form)
                    Tb = df.Function(V).vector()  # Alloc and apply
                    T.transpmult(b, Tb)
                    components.append(Tb)

            if is_trial_function(terminal):
                assert arity == 2
                replacement = df.TrialFunction(TV)
                # Passing the args to get the comparison
                integrand = replace(integrand, terminal, replacement, attributes=self.attributes)
                trace_form = Form([integral.reconstruct(integrand=integrand)])

                A = xii.assembler.xii_assembly.assemble(trace_form)
                components.append(A*T)

            # Okay, then this guy might be a function
            if isinstance(terminal, df.Function):
                replacement = df.Function(TV)
                # Replacement is not just a placeholder
                T.mult(terminal.vector(), replacement.vector())
                # Substitute
                integrand = replace(integrand, terminal, replacement, attributes=self.attributes)
                trace_form = Form([integral.reconstruct(integrand=integrand)])
                components.append(xii.assembler.xii_assembly.assemble(trace_form))

        # The whole form is then the sum of integrals
        return reduce(operator.add, components)
