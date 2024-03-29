from collections import OrderedDict, ChainMap
import dolfin as df
import numpy as np


H1_norm = lambda u, uh: df.errornorm(u, uh, 'H1', degree_rise=2)
L2_norm = lambda u, uh: df.errornorm(u, uh, 'L2', degree_rise=2)
Hdiv_norm = lambda u, uh: df.errornorm(u, uh, 'Hdiv', degree_rise=2)
Hcurl_norm = lambda u, uh: df.errornorm(u, uh, 'Hcurl', degree_rise=2)


def broken_norm(subdomains, norm_f):
    '''Evaluate norm on subdomains = {tag -> df.Subdomain}'''
    def compute(u, uh):
        # We expect u is {tag -> Expression}
        # The idea is to restict uh to marked subdomains
        V = uh.function_space()
        mesh = V.mesh()
        cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

        errors = []
        for tag in u:
            cell_f.set_all(0)
            # Restriction ...
            subdomains[tag].mark(cell_f, tag)
            sub_mesh = df.SubMesh(mesh, cell_f, tag)
            
            if not sub_mesh.num_cells(): continue
            # ... and restrict
            uh_sub = df.interpolate(uh, df.FunctionSpace(sub_mesh, V.ufl_element()))
            
            e_sub = norm_f(u[tag], uh_sub)
            errors.append(e_sub)
        # Combine errors from subdomains
        return np.sqrt(sum(e**2 for e in errors))
    # The closure
    return compute


class VarHistory(list):
    '''History where only the value is interesting'''
    def __init__(self, name, fmt=lambda s: f'{s:g}'):
        self.name = name
        self.fmt = fmt
        super(list).__init__()

    def report_last(self, with_name=True):
        '''Last value as string'''
        if with_name:
            return f'{self.name} = {self.fmt(self[-1])}'
        else:
            return f'{self.fmt(self[-1])}'

    def header(self):
        return self.name

    
class VarApproximationHistory():
    '''History where for each new value we track mesh size, approximation
    error, convergence rate and size of the finite element space.
    '''
    def __init__(self, name, u, get_error, subscript):
        self.name = name
        self.get_error = lambda uh, u=u: get_error(u, uh)
        self.errors = VarHistory(f'|e_{name}|_{subscript}', lambda s: f'{s:.3E}')
        self.rates = VarHistory(f'r|e_{name}|_{subscript}', lambda s: f'{s:.2f}')
        self.Vdims = VarHistory(f'dim(V({name}))', lambda s: f'{s:g}')
        self.hs = VarHistory(f'h(V({name}))', lambda s: f'{s:.2E}')

    def append(self, uh):
        '''Update mesh size and dims of fem space and the error + rate'''
        error = self.get_error(uh)
        self.errors.append(error)

        h = uh.function_space().mesh().hmin()
        self.hs.append(h)

        ndofs = uh.function_space().dim()
        self.Vdims.append(ndofs)

        if len(self.errors) > 1:
            try:
                rate = df.ln(self.errors[-1]/self.errors[-2])/df.ln(self.hs[-1]/self.hs[-2])
            except ZeroDivisionError:
                rate = np.nan
        else:
            rate = np.nan
        self.rates.append(rate)

    def report_last(self, with_name=True):
        return ' '.join([c.report_last(with_name)
                         for c in (self.hs, self.errors, self.rates, self.Vdims)])

    def header(self):
        return ' '.join([c.name
                         for c in (self.hs, self.errors, self.rates, self.Vdims)])

    def get_rate(self):
        '''Incremental and least squares fit'''
        incr = self.rates[-1]
        if np.isnan(incr):
            return (np.nan, np.nan)

        fit, _ = np.polyfit(np.log(self.hs), np.log(self.errors), deg=1)
        return (incr, fit) 

    
class ConvergenceLog():
    '''Monitor for error convergence studies'''
    def __init__(self, variables):
        var_histories = OrderedDict()
        avar_histories = OrderedDict()
        self.variables = tuple(variables.keys())
        
        for k, v in variables.items():
            # Separate things for error convergence from just values to track
            if v is None:
                var_histories[k] = VarHistory(k)
            else:
                u, error_f, subscript = v                
                avar_histories[k] = VarApproximationHistory(k, u, error_f, subscript)
        self.histories = ChainMap(avar_histories, var_histories)

    def __getitem__(self, key):
        return self.histories[key]

    def add(self, new):
        if not isinstance(new, dict):
            return self.add(dict(zip(self.variables, new)))

        for key in self.histories:
            self[key].append(new[key])

    def report_last(self, with_name):
        return ' '.join([format(self[key].report_last(with_name), '<20')
                         for key in self.histories])

    def header(self,):
        _header = ' '.join([self[key].header() for key in self.histories])
        return '\n'.join(['-'*len(_header), _header, '-'*len(_header)])
