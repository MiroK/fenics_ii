from . convert import convert as ii_convert
from . convert import collapse as ii_collapse
from . block_utils import (block_diag_mat, ii_PETScOperator, ii_PETScPreconditioner,
                           VectorizedOperator)
from . function import ii_Function, as_petsc_nest

try:
    from . hsmg_utils import inverse
except ImportError:
    print 'Missing HsMG for fract norm computing'
    pass

