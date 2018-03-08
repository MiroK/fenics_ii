# Driver for demos

from xii import (ii_Function, ii_assemble, ii_convert, ii_PETScOperator,
                 ii_PETScPreconditioner, as_petsc_nest)
from runpy import run_module
from dolfin import solve, File, Timer, PETScKrylovSolver
from petsc4py import PETSc
import os


def main(module_name, ncases, save_dir='', solver='direct', precond=0, eps=1., log=0):
    '''
    Run the test case in module with ncases. Optionally store results
    in savedir. For some modules there are multiple (which) choices of 
    preconditioners.
    '''
    RED = '\033[1;37;31m%s\033[0m'
    print RED % ('\tRunning %s' % module_name)

    module = __import__(module_name)  # no importlib in python2.7

    # Setup the MMS case
    u_true, rhs_data = module.setup_mms(eps)

    # Setup the convergence monitor
    if log:
        params = [('solver', solver), ('precond', str(precond)), ('eps', str(eps))]
        
        path = '_'.join([module_name] + ['%s=%s' % pv for pv in params])
        path = os.path.join(save_dir if save_dir else '.', path)
        path = '.'.join([path, 'txt'])
    else:
        path = ''

    memory = []
    monitor = module.setup_error_monitor(u_true, memory, path=path)

    # Sometimes it is usedful to transform the solution before computing
    # the error. e.g. consider subdomains
    if hasattr(module, 'setup_transform'):
        # NOTE: transform take two args for case and the current computed
        # solution
        transform = module.setup_transform
    else:
        transform = lambda i, x: x

    print '='*79
    print '\t\t\tProblem eps = %g' % eps
    print '='*79
    for i in ncases:
        a, L, W = module.setup_problem(i, rhs_data, eps=eps)

        # Assemble blocks
        t = Timer('assembly'); t.start()
        AA, bb = map(ii_assemble, (a, L))
        print '\tAssembled blocks in %g s' % t.stop()

        wh = ii_Function(W)
        
        if solver == 'direct':
            # Turn into a (monolithic) PETScMatrix/Vector
            t = Timer('conversion'); t.start()        
            AAm, bbm = map(ii_convert, (AA, bb))
            print '\tConversion to PETScMatrix/Vector took %g s' % t.stop()
            
            t = Timer('solve'); t.start()
            solve(AAm, wh.vector(), bbm)
            print '\tSolver took %g s' % t.stop()
            
            niters = 1
        else:
            # Here we define a Krylov solver using PETSc
            BB = module.setup_preconditioner(W, precond, eps=eps)
            ## AA and BB as block_mat
            ksp = PETSc.KSP().create()
            ksp.setType('minres')
            ksp.setOperators(ii_PETScOperator(AA))

            def ksp_monitor(ksp, k, norm):
                print 'iteration %d norm %g' % (k, norm)

            ksp.setMonitor(ksp_monitor)

            ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
            ksp.setTolerances(rtol=1E-8, atol=None, divtol=None, max_it=200)
            ksp.setConvergenceHistory()
            # We attach the wrapped preconditioner defined by the module
            ksp.setPC(ii_PETScPreconditioner(BB, ksp))

            # Want the iterations to start from random
            wh.block_vec().randomize()
            
            # Solve, note the past object must be PETSc.Vec
            t = Timer('solve'); t.start()            
            ksp.solve(as_petsc_nest(bb), wh.petsc_vec())
            print '\tSolver took %g s' % t.stop()

            niters = ksp.getIterationNumber()
            
        # Let's check the final size of the residual
        r_norm = (bb - AA*wh.block_vec()).norm()
            
        # Convergence?
        monitor.send((transform(i, wh), niters, r_norm))
        
    # Only send the final
    if save_dir:
        path = os.path.join(save_dir, module_name)
        for i, wh_i in enumerate(wh):
            # Renaming to make it easier to save state in Visit/Pareview
            wh_i.rename('u', str(i))
            File('%s_%d.pvd' % (path, i)) << wh_i

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The demo file: runnning it defines setups*
    parser.add_argument('demo', nargs='?', default='all', type=str,
                        help='Which demo to run')
    # Some problems are paramer dependent
    parser.add_argument('-problem_eps', default=[1.0], type=float, nargs='+',
                        help='Parameter value for problem setup')
    # How many uniform refinements to make
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')
    # How many uniform refinements to make
    parser.add_argument('-save_dir', type=str, default='',
                        help='Path for directory storing results')
    # Log the results
    parser.add_argument('-log', type=int, default=0,
                        help='0/1 for storing the results')

    # Iterative solver?
    parser.add_argument('-solver', type=str, default='direct', choices=['direct', 'iterative'],
                        help='Use direct solver with params as in main')
    # Choice of preconditioner
    parser.add_argument('-precond', type=int, default=0,
                        help='Which module preconditioner')

    args = parser.parse_args()
    assert args.ncases > 0
    assert all(e > 0 for e in args.problem_eps)

    if args.save_dir:
        assert os.path.exists(args.save_dir)
        assert os.path.isdir(args.save_dir)

    # The setups
    module, _ = os.path.splitext(args.demo)

    if module == 'all':
        modules = [os.path.splitext(f)[0] for f in os.listdir('.') if f.endswith('_2d.py')]
    else:
        modules = [module]

    for module in modules:
        for e in args.problem_eps:
            main(module, range(args.ncases), save_dir=args.save_dir,
                                             solver=args.solver,
                                             precond=args.precond,
                                             eps=e,
                                             log=bool(args.log))
