import subprocess 

GREEN = '\033[1;37;32m%s\033[0m'
RED = '\033[1;37;31m%s\033[0m'


def run(cmd, *options):
    '''Run the demo convergence study'''
    cmd = ['python', cmd] + list(options)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

    return p.returncode == 1


demos = (('poisson_babuska.py', '--conformity', 'nested'),
         ('poisson_babuska.py', '--conformity', 'non_nested'),
         ('poisson_babuska.py', '--conformity', 'conforming'),
         ('poisson_babuska.py', '--conformity', 'conforming_facetf'))


results = {True: GREEN % 'Passed', False: RED % 'Failed'}
for demo in demos:
    cmd, *options = demo
    status = run(cmd, *options)
    print(f'Ran `{" ".join([cmd]+list(options))}` .... {results[status]}')
