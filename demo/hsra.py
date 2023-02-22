from block.algebraic.hazmath import RA
import haznics


def HsRA(A, M, a0, s0, ra_tol, a1=0, s1=0):
    '''RA realization of H^s norm'''
    # parameters for RA and AMG
    params = {'coefs': [a0, a1], 'pwrs': [s0, s1],  # for RA
              'print_level': 10,
              'AMG_type': haznics.SA_AMG,
              'cycle_type': haznics.V_CYCLE,
              "max_levels": 20,
              "tol": 1E-10,
              'AAA_tol': ra_tol,
              "smoother": haznics.SMOOTHER_GS,
              "relaxation": 1.2,  # Relaxation in the smoother
              "coarse_dof": 10,
              "aggregation_type": haznics.VMB,  # (VMB, MIS, MWM, HEC)
              "strong_coupled": 0.0,  # threshold
              "max_aggregation": 100
              }

    precond = RA(A, M, parameters=params)
    
    return precond
