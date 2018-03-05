# Driver for demos

from runpy import run_module
from xii import ii_Function
from dolfin import solve, File
import os


def main(module_name, ncases, save_dir=''):
    '''Run the test case in module with ncases'''
    RED = '\033[1;37;31m%s\033[0m'
    print RED % ('\tRunning %s' % module_name)

    module = __import__(module_name)  # no importlib in python2.7

    # Setup the MMS case
    u_true, rhs_data = module.setup_mms()

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
    
    for i in ncases:
        AA, bb, W = module.solve_problem(i, rhs_data)

        wh = ii_Function(W)
        solve(AA, wh.vector(), bb)
        monitor.send(transform(i, wh))
        
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
    # How many uniform refinements to make
    parser.add_argument('-ncases', type=int, default=1,
                        help='Run convergence study with # cases')
    # How many uniform refinements to make
    parser.add_argument('-save_dir', type=str, default='',
                        help='Path for directory storing results')

    args = parser.parse_args()
    assert args.ncases > 0

    if args.save_dir:
        assert os.path.exists(args.save_dir)
        assert os.path.isdir(args.save_dir)

    # The setups
    module, _ = os.path.splitext(args.demo)

    if module == 'all':
        modules = [os.path.splitext(f)[0] for f in os.listdir('.') if f.endswith('_2d.py')]
    else:
        modules = [module]

    [main(module, range(args.ncases), args.save_dir) for module in modules]
