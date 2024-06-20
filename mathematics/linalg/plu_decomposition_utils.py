'''
Helper functions for functions in `plu_decomposition`.

Source:  https://en.m.wikipedia.org/wiki/LU_decomposition#C_code_example
'''

from .matrix import zeroes_square, matmul

from copy import deepcopy

def LUPDecompose(A, N, tolerance):
    '''Helper function for PLU decomposition
    
    Input:
    
     - `A`: matrix with dimensions `N` by `N`
     - `N`: integer
     - `tolerance`: threshold for considering a float equal to 0
    
    Output: (status, P), where

     - `status`: `1` if matrix successfully decomposed (non-degenerate), `0` otherwise
     - `P` is a vector of length `N+1` representing the permutation matrix of the LU decomposition; specifically:

        - for `i < N`, `P[i]` is the column in the `i`th row that contains a `1` in the permutation matrix
        - `P[N]` = `S + N`, where `S` is the number of row exchanges needed for the computation of \\(\\det(A)\\); that is, \\(\\det(P) = (-1)^S\\)
    
    Side effects:

     - Modifies `A` to be \\((L - I) + U\\), where:

        - \\(L\\) and \\(U\\) correspond to the LU factorization of `A`
        - \\(I\\) is the `N`-dimensional identity matrix
    '''
    P = list(range(N + 1))
    for i in range(N):
        maxA = 0.0
        imax = i
        for k in range(i, N):
            absA = abs(A[k][i])
            if absA > maxA:
                maxA = absA
                imax = k
        if maxA < tolerance:
            return 0, P  # Failure, matrix is degenerate
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            A[i], A[imax] = A[imax], A[i]
            P[N] += 1
        for j in range(i + 1, N):
            A[j][i] /= A[i][i]
            for k in range(i + 1, N):
                A[j][k] -= A[j][i] * A[i][k]
    return 1, P  # Decomposition done

def LUPSolve(A, P, b, N):
    '''Helper function for solving \\(A\\vec x = B\\) using PLU decomposition'''
    x = [0] * N
    for i in range(N):
        x[i] = b[P[i]]
        for k in range(i):
            x[i] -= A[i][k] * x[k]
    for i in range(N - 1, -1, -1):
        for k in range(i + 1, N):
            x[i] -= A[i][k] * x[k]
        x[i] /= A[i][i]
    return x

def LUPInvert(A, P, N):
    '''Helper function for inverting a matrix using PLU decomposition'''
    IA = [[0] * N for _ in range(N)]
    for j in range(N):
        for i in range(N):
            IA[i][j] = 1.0 if P[i] == j else 0.0
            for k in range(i):
                IA[i][j] -= A[i][k] * IA[k][j]
        for i in range(N - 1, -1, -1):
            for k in range(i + 1, N):
                IA[i][j] -= A[i][k] * IA[k][j]
            IA[i][j] /= A[i][i]
    return IA

def extractLU(A, N):
    '''Extracts matrices `L` and `U` from `A`. 
    
    `A` should be the result of LUPDecompose(A, N, tolerance)'''
    L = [[0] * N for _ in range(N)]
    U = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i > j:
                L[i][j] = A[i][j]
            elif i == j:
                L[i][j] = 1
                U[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
    return L, U

def expandP(P):
    '''Helper function for converting the `P` vector returned by LUPDecompose back into a permutation matrix'''
    z = zeroes_square(len(P)-1)
    for i,e in enumerate(P[:-1]):
        z[i][e] = 1
    return z


if __name__ == "__main__":
    N = 4
    tolerance = 1e-9

    A = [[1, 3, 1, 4],[3, 9, 5, 15],[0, 2, 1, 1],[0, 4, 2, 3]]
    A_init = deepcopy(A)
    b = [4, 2, 5, 3]

    # test LUPDecompose
    status, P = LUPDecompose(A, N, tolerance)
    assert status == 1, 'Failed test: LUPDecompose'

    # test LUPSolve
    x = LUPSolve(A, P, b, N)
    expected = [16.75, 3.2499999999999996, 5.500000000000001, -7.0]
    assert x == expected, 'Failed test: LUPSolve'

    # test LUPInvert
    IA = LUPInvert(A, P, N)
    expected = [
        [0.25, 0.25, 5.0, -3.25],
        [0.7500000000000002, -0.25000000000000006, -4.440892098500626e-16, 0.2500000000000002],
        [-1.5000000000000004, 0.5000000000000001, 3.000000000000001, -1.5000000000000004],
        [-0.0, -0.0, -2.0, 1.0]
    ]
    assert IA == expected, 'Failed test: LUPInvert'

    # test extractLU and expandP
    L, U = extractLU(A, N)
    P = expandP(P)
    assert matmul(L, U) == matmul(P, A_init), 'Failed tests: extractLU, expandP'

    print('Passed all tests.')