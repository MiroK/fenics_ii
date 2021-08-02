import subprocess 
import pytest


def run(cmd, *options):
    '''Run the demo convergence study'''
    cmd = ['python', cmd] + list(options)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = p.communicate()

    if 'Error' in out.decode():
        return 0
    
    return p.returncode == 1


demos = (('poisson_babuska.py', '--conformity', 'nested'),
         ('poisson_babuska.py', '--conformity', 'non_nested'),
         ('poisson_babuska.py', '--conformity', 'conforming'),
         ('poisson_babuska.py', '--conformity', 'conforming_facetf'),
         #
         ('poisson_babuska_bc.py', '--bcs', 'dir_neu'),
         ('poisson_babuska_bc.py', '--bcs', 'dir'),
         ('poisson_babuska_bc.py', '--bcs', 'neu'),
         #
         ('mixed_poisson_babuska.py', '--flux_degree', '1'),
         ('mixed_poisson_babuska.py', '--flux_degree', '2'),
         #
         ('poisson_babuska_3d.py', ),
         #
         ('grad_div_babuska.py', '--RT_degree', '1'),
         ('grad_div_babuska.py', '--RT_degree', '2'),
         #
         ('curl_curl_babuska.py', '--Ned_degree', '1'),
         ('curl_curl_babuska.py', '--Ned_degree', '2'),
         #
         ('sym_grad_babuska.py', '--Bop', 'full'),
         ('sym_grad_babuska.py', '--Bop', 'normal'),
         ('sym_grad_babuska.py', '--Bop', 'tangent'),
         #
         ('bertoluzza.py', '--is_flat', '1'),
         ('bertoluzza.py', '--is_flat', '0'),
         #
         ('dq_darcy_stokes.py', ),  # With default discretization and unit params
         ('dq_darcy_stokes.py', '--param_mu', '0.5', '--param_K', '2', '--param_alpha', '1'),
         ('dq_darcy_stokes.py', '--param_mu', '0.5', '--param_K', '2', '--param_alpha', '0'),
         ('dq_darcy_stokes.py', '--pS_degree', '1', '--pD_degree', '1'),
         ('dq_darcy_stokes.py', '--pS_degree', '0', '--pD_degree', '1'),
         ('dq_darcy_stokes.py', '--pS_degree', '0', '--pD_degree', '2'),
         #
         ('layton_darcy_stokes.py', ),  # With default (TH Stokes) discretization and unit params
         ('layton_darcy_stokes.py', '--param_mu', '0.5', '--param_K', '4', '--param_alpha', '1'),
         ('layton_darcy_stokes.py', '--param_mu', '0.5', '--param_K', '4', '--param_alpha', '0'),
         ('layton_darcy_stokes.py', '--param_mu', '0.5', '--param_K', '4', '--param_alpha', '1', '--stokes_CR', '1'), # Try with stabilized Crouzeix-Raviart
         #
         ('emi_primal.py', ),
         ('emi_primal.py', '--param_kappa0', '2', '--param_kappa1', '4'),
         ('emi_primal.py', '--degree', '2', '--param_kappa0', '3', '--param_kappa1', '5'),
         #
         ('emi_primal_mortar.py', ),
         ('emi_primal_mortar.py', '--param_kappa0', '2', '--param_kappa1', '4'),
         ('emi_primal_mortar.py', '--degree', '2', '--param_kappa0', '3', '--param_kappa1', '5'),         
         #
         ('twoDoneDoneD.py', ),
         ('twoDoneDoneD.py', '--param_kappa', '2', '--param_kappa1', '3'),
)


@pytest.mark.parametrize('args', demos)
def test_demos(args):
    cmd, *options = args
    assert run(cmd, *options)
    
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    GREEN = '\033[1;37;32m%s\033[0m'
    RED = '\033[1;37;31m%s\033[0m'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Decide material parameters
    parser.add_argument('--selected', type=str, nargs='+', default=[], help='Stokes viscosity')
    args, _ = parser.parse_known_args()

    if args.selected:
        demos = tuple(d for d in demos if d[0] in args.selected)
    
    results = {True: GREEN % 'Passed', False: RED % 'Failed'}
    for demo in demos:
        cmd, *options = demo
        status = run(cmd, *options)
        print(f'Ran `{" ".join([cmd]+list(options))}` .... {results[status]}')
