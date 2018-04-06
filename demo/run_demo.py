# Driver for demos

from xii import (ii_Function, ii_assemble, ii_convert, ii_PETScOperator,
                 ii_PETScPreconditioner, as_petsc_nest)
from runpy import run_module
from dolfin import solve, File, Timer, LUSolver, interpolate
import matplotlib.pyplot as plt

import os

def main(module_name, ncases, params, petsc_params):
    '''
    Run the test case in module with ncases. Optionally store results
    in savedir. For some modules there are multiple (which) choices of 
    preconditioners.
    '''
    
    # Unpack
    for k, v in params.items(): exec(k + '=v', locals())

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

    memory, residuals  = [], []
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
            LUSolver('umfpack').solve(AAm, wh.vector(), bbm)
            print '\tSolver took %g s' % t.stop()
            
            niters = 1
        else:
            # Here we define a Krylov solver using PETSc
            BB = module.setup_preconditioner(W, precond, eps=eps)
            ## AA and BB as block_mat
            ksp = PETSc.KSP().create()
            ksp.setType('minres')
            ksp.setOperators(ii_PETScOperator(AA))

            ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
            # ksp.setTolerances(rtol=1E-6, atol=None, divtol=None, max_it=300)
            ksp.setConvergenceHistory()
            # We attach the wrapped preconditioner defined by the module
            ksp.setPC(ii_PETScPreconditioner(BB, ksp))
            
            opts = PETSc.Options()
            for key, value in petsc_params.iteritems():
                opts.setValue(key, None if value == 'none' else value)
            ksp.setFromOptions()
            
            print ksp.getTolerances()
            
            # Want the iterations to start from random
            wh.block_vec().randomize()
            # Solve, note the past object must be PETSc.Vec
            t = Timer('solve'); t.start()            
            ksp.solve(as_petsc_nest(bb), wh.petsc_vec())
            print '\tSolver took %g s' % t.stop()

            niters = ksp.getIterationNumber()

            residuals.append(ksp.getConvergenceHistory())
            
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

    # Plot relative residual norm
    if plot:
        plt.figure()
        [plt.semilogy(res/res[0], label=str(i)) for i, res in enumerate(residuals, 1)]
        plt.legend(loc='best')
        plt.show()

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys, petsc4py
    # Make petsc4py work with command line. This allows configuring
    # ksp (petsc krylov solver) as e.g.
    #      -ksp_rtol 1E-8 -ksp_atol none -ksp_monitor_true_residual none
    petsc4py.init(sys.argv)
    from petsc4py import PETSc

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
    # Plot iters vs residual norm
    parser.add_argument('-plot', type=int, default=0,
                        help='0/1 for plotting iters vs. residual norm')

    # Iterative solver?
    parser.add_argument('-solver', type=str, default='direct', choices=['direct', 'iterative'],
                        help='Use direct solver with params as in main')
    # Choice of preconditioner
    parser.add_argument('-precond', type=int, default=0,
                        help='Which module preconditioner')

    args, petsc_args = parser.parse_known_args()

    if petsc_args:
        petsc_args = dict((k, v) for k, v in zip(petsc_args[::2], petsc_args[1::2]))
    
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

    params = {'save_dir': args.save_dir, 'solver': args.solver,
              'precond': args.precond, 'log': bool(args.log), 'plot': bool(args.plot)}
    for module in modules:
        for e in args.problem_eps:
            params['eps'] = e
            main(module, ncases=range(args.ncases), params=params, petsc_params=petsc_args)
