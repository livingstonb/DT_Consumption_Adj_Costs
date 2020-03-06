import numpy as np

def ergodicdist(Q):
    # ErgodicDist - Ergodic distribution of discrete Markov Chains
    #
    # q=ergodicdist(Q)
    #
    # Input: Q -   right stochastic matrix (COLUMNS sum to one): the prob.
    #              of reaching state j from state i is P(i,j).
    #        alg - choice of method: 1: iterative, 2: direct. (optional;
    #              default = 1)
    #
    # Output: q - (nx1) vector containing the ergodic distribution
    #
    # Author: Marco Maffezzoli. Ver. 1.0.1, 11/2012.

    h = Q.shape[0]
    q = np.zeros((1, h))
    q[0,0] = 1

    dif = 1
    while dif > 1e-8:
        z = np.matmul(q, Q)
        dif = np.linalg.norm(z-q)
        q = z

    q = np.transpose(q)
    return q