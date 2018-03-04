# Driver for demos

from runpy import run_module


def main(module, ncells):
    '''Run the test case in module with ncells'''
    RED = '\033[1;37;31m%s\033[0m'
    print RED % ('\tRuning %s' % module)

    module = __import__(module)  # no importlib in python2.7

    # Setup the MMS case
    u_true, rhs_data = module.setup_mms()

    # Setup the convergence monitor
    memory = []
    monitor = module.setup_error_monitor(u_true, memory)
    
    for n in ncells:
        result = module.solve_problem(n, rhs_data)
        monitor.send(result)

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # The demo file: runnning it defines setups*
    parser.add_argument('demo', nargs='?', default='all', type=str, help='Which demo to run')
    # Seed the refinements
    parser.add_argument('-seed', type=int, default=2, help='Resolutions to start with')
    # How many uniform refinements to make
    parser.add_argument('-nrefines', type=int, default=5, help='How many times to refine')
    # How many uniform refinements to make
    parser.add_argument('-save', type=str, default='', help='Path for storing')

    args = parser.parse_args()
    assert args.seed > 0 and args.nrefines > -1

    ncells = [args.seed] 
    for i in range(args.nrefines): ncells.append(2*ncells[-1])

    # The setups
    module, _ = os.path.splitext(args.demo)

    if module == 'all':
        modules = [os.path.splitext(f)[0] for f in os.listdir('.') if f.endswith('_2d.py')]
    else:
        modules = [module]

    [main(module, ncells) for module in modules]
