'''
PLU decomposition and matrix operations that benefit from it.

See `plu_decomposition_utils` for underlying implementation.

Source:  https://en.m.wikipedia.org/wiki/LU_decomposition#C_code_example
'''

from .matrix import diag, matmul
from .plu_decomposition_utils import *

from copy import deepcopy
from math import prod

def PLUDecomposition(A, tolerance=1e-9):
    '''PLU decomposition of a matrix `A`
    
    Input:
    
     - `A`: an \\(n \\times n\\) matrix
     - `tolerance`: threshold for considering a float equal to 0. Defaults to `1e-9`

    Output:

     - `P`: permutation matrix
     - `L`: lower triangular matrix
     - `U`: upper triangular matrix

    `P`, `L`, and `U` are guaranteed to obey \\(PA = LU\\).

    Side effects:

     - Modifies `A` in place
    
    If `A` is singular/degenerate this will throw.'''
    if len(A) != len(A[0]):
        raise ValueError('expected a square matrix, got dimensions ({},{})'.format(len(A), len(A[0])))
    N = len(A)
    status, P = LUPDecompose(A, N, tolerance)
    if status == 0:
        raise ArithmeticError('degenerate matrix detected')
    L, U = extractLU(A, N)
    P = expandP(P)
    return P, L, U

def PLUDeterminant(A, tolerance=1e-9):
    '''Determinant of a \\(n \\times n\\) matrix `A` using PLU decomposition
    
    Input:
    
     - `A`: an \\(n \\times n\\) matrix
     - `tolerance`: threshold for considering a float equal to 0. Defaults to `1e-9`

    Output:

     - `d`: \\(\\det A\\)

    Side effects:

     - Modifies `A` in place
    
    If `A` is singular/degenerate this will throw.'''
    if len(A) != len(A[0]):
        raise ValueError('expected a square matrix, got dimensions ({},{})'.format(len(A), len(A[0])))
    N = len(A)
    status, P = LUPDecompose(A, N, tolerance)
    if status == 0:
        raise ArithmeticError('degenerate matrix detected')
    L, U = extractLU(A, N)
    return prod(diag(L)) * prod(diag(U)) * (-1 if (P[N] - N) % 2 else 1)

def PLUSolve(A, b, tolerance=1e-9):
    '''Solution \\(\\vec x\\) to \\(A\\vec x = b\\) using PLU decomposition
    
    Input:
    
     - `A`: an \\(n \\times n\\) matrix
     - `b`: a list (vector) of length \\(n\\)
     - `tolerance`: threshold for considering a float equal to 0. Defaults to `1e-9`

    Output:

     - `x`: solution vector to \\(A\\vec x = b\\)

    Side effects:

     - Modifies `A` in place
    
    If `A` is singular/degenerate this will throw.'''
    if len(A) != len(A[0]):
        raise ValueError('expected a square matrix, got dimensions ({},{})'.format(len(A), len(A[0])))
    N = len(A)
    status, P = LUPDecompose(A, N, tolerance)
    if status == 0:
        raise ArithmeticError('degenerate matrix detected')
    return LUPSolve(A, P, b, N)

def PLUInvert(A, tolerance=1e-9):
    '''Invert a matrix `A` using PLU decomposition
    
    Input:
    
     - `A`: an \\(n \\times n\\) matrix
     - `tolerance`: threshold for considering a float equal to 0. Defaults to `1e-9`

    Output:

     - \\(A^{-1}\\): inverted version of `A`

    Side effects:

     - Modifies `A` in place
    
    If `A` is singular/degenerate this will throw.'''
    if len(A) != len(A[0]):
        raise ValueError('expected a square matrix, got dimensions ({},{})'.format(len(A), len(A[0])))
    N = len(A)
    status, P = LUPDecompose(A, N, tolerance)
    if status == 0:
        raise ArithmeticError('degenerate matrix detected')
    return LUPInvert(A, P, N)

if __name__ == "__main__":
    N = 4
    tolerance = 1e-9

    # Initialize matrix A
    A = [[1, 3, 1, 4],[3, 9, 5, 15],[0, 2, 1, 1],[0, 4, 2, 3]]
    A_init = deepcopy(A)
    b = [4, 2, 5, 3]

    # Perform LUP Decomposition
    status, P = LUPDecompose(A, N, tolerance)
    assert status == 1, 'Failed test: LUPDecompose'

    # Solve the system A*x = b
    x = LUPSolve(A, P, b, N)
    expected = [16.75, 3.2499999999999996, 5.500000000000001, -7.0]
    assert x == expected, 'Failed test: LUPSolve'

    x = PLUSolve(deepcopy(A_init), b)
    expected = [16.75, 3.2499999999999996, 5.500000000000001, -7.0]
    assert x == expected, 'Failed test: PLUSolve'

    L, U = extractLU(A, N)
    P = expandP(P)
    assert matmul(L, U) == matmul(P, A_init), 'Failed tests: extractLU, expandP, PLUDecomposition'
    
    det = PLUDeterminant(deepcopy(A_init))
    expected = -3.999999999999999
    assert det == expected, 'Failed test: PLUDeterminant'

    # Find the inverse of the matrix
    IA = PLUInvert(deepcopy(A_init))
    expected = [
        [0.25, 0.25, 5.0, -3.25],
        [0.7500000000000002, -0.25000000000000006, -4.440892098500626e-16, 0.2500000000000002],
        [-1.5000000000000004, 0.5000000000000001, 3.000000000000001, -1.5000000000000004],
        [-0.0, -0.0, -2.0, 1.0]
    ]
    assert IA == expected, 'Failed test: PLUInvert'

    print('Passed all tests.')