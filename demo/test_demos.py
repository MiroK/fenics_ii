import subprocess 
import pytest

def run(cmd, *options):
    '''Run the demo convergence study'''
    cmd = ['python', cmd] + list(options)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

    return p.returncode == 1

demos = (('poisson_babuska.py', '--conformity', 'nested'),
         ('poisson_babuska.py', '--conformity', 'non_nested'),
         ('poisson_babuska.py', '--conformity', 'conforming'),
         ('poisson_babuska.py', '--conformity', 'conforming_facetf'),
         ('poisson_babuska_bc.py', '--bcs', 'dir_neu'),
         ('poisson_babuska_bc.py', '--bcs', 'dir'),
         ('poisson_babuska_bc.py', '--bcs', 'neu'),
         ('mixed_poisson_babuska.py', '--flux_degree', '1'),
         ('mixed_poisson_babuska.py', '--flux_degree', '2'),
         ('poisson_babuska_3d.py', ),
)


@pytest.mark.parametrize('args', demos)
def test_demos(args):
    cmd, *options = args
    assert run(cmd, *options)
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    GREEN = '\033[1;37;32m%s\033[0m'
    RED = '\033[1;37;31m%s\033[0m'
    
    results = {True: GREEN % 'Passed', False: RED % 'Failed'}
    for demo in demos:
        cmd, *options = demo
        status = run(cmd, *options)
        print(f'Ran `{" ".join([cmd]+list(options))}` .... {results[status]}')
