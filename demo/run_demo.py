# Driver for demos

from runpy import run_module
from xii import ii_Function, ii_assemble, ii_convert
from dolfin import solve, File, Timer
import os


def main(module_name, ncases, save_dir='', solver='direct', precond=0, eps=1.):
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
    memory = []
    monitor = module.setup_error_monitor(u_true, memory)

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

        if solver == 'direct':
            # Turn into a (monolithic) PETScMatrix/Vector
            t = Timer('conversion'); t.start()        
            AAm, bbm = map(ii_convert, (AA, bb))
            print '\tConversion to PETScMatrix/Vector took %g s' % t.stop()
            
            wh = ii_Function(W)
            t = Timer('solve'); t.start()
            solve(AAm, wh.vector(), bbm)
            print '\tSolver took %g s' % t.stop()
            
            niters = None  # Well 1
        else:
            from block.iterative import PETScMinRes

            # Start from random initial guess
            x = AA.create_vec(); x.randomize()
            tolerance = 1E-8
            relativeconv = True
            
            BB = module.setup_preconditioner(W, precond, eps=eps)
            
            AAinv = PETScMinRes(AA, precond=BB, initial_guess=x,
                                tolerance=tolerance,
                                relativeconv=relativeconv)
            # Solve
            t = Timer('solve'); t.start()
            x = AAinv*bb
            print '\tSolver took %g s' % t.stop()
            
            niters = len(AAinv.residuals)-1
            
            wh = ii_Function(W, x)
        # Let's check the final size of the residual
        r_norm = (bb - AA*wh.block_vec()).norm()
            
        # Convergence?
        monitor.send((transform(i, wh), niters, r_norm))
        
    # Only send the final
    if save_dir:
        path = os.path.join(save_dir, module_name)
        for i, wh_i in enumerate(wh):
            File('%s_%d.pvd' % (path, i)) << wh_i

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The demo file: runnning it defines setups*
    parser.add_argument('demo', nargs='?', default='all', type=str,
                        help='Which demo to run')
    # Some problems are paramer dependent
    parser.add_argument('-problem_eps', default=1.0, type=float,
                        help='Paramter value for problem setup')
    # How many uniform refinements to make
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')
    # How many uniform refinements to make
    parser.add_argument('-save_dir', type=str, default='',
                        help='Path for directory storing results')

    # Iterative solver?
    parser.add_argument('-solver', type=str, default='direct', choices=['direct', 'iterative'],
                        help='Use direct solver with params as in main')
    # Choice of preconditioner
    parser.add_argument('-precond', type=int, default=0,
                        help='Which module preconditioner')

    args = parser.parse_args()
    assert args.ncases > 0
    assert args.problem_eps > 0

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
        main(module, range(args.ncases), save_dir=args.save_dir,
                                         solver=args.solver,
                                         precond=args.precond,
                                         eps=args.problem_eps)
