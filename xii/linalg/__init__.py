from . convert import convert as ii_convert
from . convert import collapse as ii_collapse
from . convert import set_lg_map
from . block_utils import (block_diag_mat, ii_PETScOperator, ii_PETScPreconditioner,
                           VectorizedOperator, ReductionOperator, BlockPC)
from . function import ii_Function, as_petsc_nest
from . bc_apply import apply_bc

try:
    from . hsmg_utils import inverse
except ImportError:
    print 'Missing HsMG for fract norm computing'
    pass

