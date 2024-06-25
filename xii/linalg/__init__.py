from . collapse_iter import monolithic
from . convert import convert as ii_convert
from . convert import collapse as ii_collapse
from . convert import set_lg_map
from . block_utils import (block_diag_mat, ii_PETScOperator, ii_PETScPreconditioner,
                           VectorizedOperator, ReductionOperator, BlockPC, SubSelectOperator)
from . function import ii_Function, as_petsc_nest
from . bc_apply import apply_bc
from . block_nullspace import BlockNullspace
from . nest import nest, pc_nest
# from . operators import OuterProduct

try:
    from . hsmg_utils import inverse
except ImportError:
    print('Missing HsMG for fract norm computing')
    pass

