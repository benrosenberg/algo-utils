'''
Matrix utilities.
'''

from math import log2
from copy import deepcopy
from functools import reduce

def zeroes(n, m):
    '''A zero matrix of dimension \\(n\\times m\\)'''
    if n <= 0 or m <= 0: raise ValueError('expecting n >= 1 and m >= 1')
    return [[0 for _ in range(m)] for _ in range(n)]

def zeroes_square(n):
    '''A zero matrix of dimension \\(n\\times n\\)'''
    if n <= 0: raise ValueError('expecting n >= 1')
    return [[0 for _ in range(n)] for _ in range(n)]

def identity(n):
    '''An identity matrix of dimension \\(n\\times n\\)'''
    if n <= 0: raise ValueError('expecting n >= 1')
    z = zeroes_square(n)
    for i in range(n):
        z[i][i] = 1
    return z

def matmul_square(A, B):
    '''Multiply square matrices `A` and `B` (naively)'''
    if len(A) != len(A[0]):
        raise ValueError('matrix A is not square ({},{})'.format(len(A), len(A[0])))
    if len(B) != len(B[0]):
        raise ValueError('matrix B is not square ({},{})'.format(len(B), len(B[0])))
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError('dimension mismatch between A () and B (), expected identical dimensions'.format((len(A), len(A[0])), (len(B), len(B[0]))))
    n = len(A)
    C = zeroes_square(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matmul(A, B):
    '''Multiply matrices `A` and `B` (naively)'''
    an, am = len(A), len(A[0])
    bn, bm = len(B), len(B[0])
    if am != bn: 
        raise ValueError('unable to multiply matrices of dimensions ({}),({}): misaligned dimensions ({} != {})'.format((an, am), (bn, bm), am, bn))
    if an == am == bn == bm:
        return matmul_square(A, B)
    n = an
    m = am
    p = bm
    C = zeroes(n, p)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C
    
def matpow(A, n):
    '''Raise matrix `A` to integer power `n` (with \\(\\log(n)\\) multiplications)'''
    if n < 0:
        raise ValueError('expected n >= 0')
    if int(n) != n:
        raise ValueError('expected integer n, got {}'.format(n))
    if len(A) != len(A[0]):
        raise ValueError('expected square matrix, got dimensions ({}, {})'.format(len(A), len(A[0])))
    if n == 0: return identity(n)
    if n == 1: return A
    acc = deepcopy(A)
    log_iters = int(log2(n))
    extra_results = []
    for i in range(log_iters+1):
        # do we need to cache this result?
        if n & (1 << i):
            extra_results.append(deepcopy(acc))
        if i < log_iters: 
            # only update acc when necessary
            acc = matmul_square(acc, acc)
    # multiply all cached results together
    return reduce(matmul_square, extra_results)

def transpose(A):
    '''Transpose of a matrix `A`'''
    return list(zip(*A))

def col(A, i):
    '''Get the `i`th column of a matrix `A`'''
    return [row[i] for row in A]

def col_range(A, i=None, j=None):
    '''Columns `i`..`j` of a matrix `A` (either `i` or `j` optional)'''
    if i is None and j is None: 
        raise ValueError('expecting non-None value for either i or j')
    if i is None:
        return [row[:j+1] for row in A]
    elif j is None:
        return [row[i:] for row in A]
    else:
        return [row[i:j+1] for row in A]

def row_range(A, i=None, j=None):
    '''Rows `i`..`j` of a matrix `A` (either `i` or `j` optional)'''
    if i is None and j is None: 
        raise ValueError('expecting non-None value for either i or j')
    if i is None:
        return A[:j+1]
    elif j is None:
        return A[i:]
    else:
        return A[i:j+1]
    
def diag(A):
    '''Diagonal elements of a matrix `A`'''
    n = min(len(A), len(A[0]))
    return [A[i][i] for i in range(n)]

if __name__ == '__main__':
    # test matpow
    A = [
        [1, 2, 0, 5, 0],
        [4, 0, 0, 4, 0],
        [0, 2, 1, 0, 7],
        [0, 0, 0, 0, 0],
        [8, 0, 54, 0, 5]
    ]
    n = 5
    expected = [
        [225, 178, 0, 581, 0],
        [356, 136, 0, 628, 0],
        [314048, 368210, 1930825, 158720, 1264963],
        [0, 0, 0, 0, 0],
        [1655976, 588752, 9758286, 411872, 2653661]
    ]
    assert matpow(A, n) == expected, 'Failed test: matpow'

    # test matmul
    A = [
        [1, 2, 3],
        [4, 3, 5]
    ]
    B = [
        [1, 3, 0, 3],
        [4, 2, 5, 0],
        [0, 6, 4, 6]
    ]
    expected = [
        [9, 25, 22, 21],
        [16, 48, 35, 42]
    ]
    assert matmul(A, B) == expected, 'Failed test: matmul'

    # test matmul_square
    A = [
        [1, 2, 4, 7],
        [0, 3, 0, 6],
        [5, 0, 4, 5],
        [0, 6, 0, 0]
    ]
    B = [
        [0, 3, 0, 3],
        [0, 4, 2, 0],
        [8, 5, 1, 2],
        [6, 5, 0, 3]
    ]
    expected = [
        [74, 66, 8, 32],
        [36, 42, 6, 18],
        [62, 60, 4, 38],
        [0, 24, 12, 0]
    ]
    assert matmul_square(A, B) == expected, 'Failed test: matmul_square'

    # test identity
    n = 3
    expected = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert identity(n) == expected, 'Failed test: identity'

    # test zeroes
    n = 3
    m = 4
    expected = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    assert zeroes(n, m) == expected, 'Failed test: zeroes'

    # test zeroes_square
    n = 3
    expected = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    assert zeroes_square(n) == expected, 'Failed test: zeroes_square'

    # test transpose
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    expected = [
        (1, 4),
        (2, 5),
        (3, 6)
    ]
    assert transpose(A) == expected, 'Failed test: transpose'

    # test col
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    i = 0
    expected = [1, 4]
    assert col(A, i) == expected, 'Failed test: col'

    # test col_range
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    i = 1
    j = 2
    expected = [
        [2, 3],
        [5, 6],
        [8, 9]
    ]
    assert col_range(A, i, j) == expected, 'Failed test: col_range'

    # test row_range
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    i = 1
    j = 2
    expected = [
        [4, 5, 6],
        [7, 8, 9]
    ]
    assert row_range(A, i, j) == expected, 'Failed test: row_range'

    # test diag
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    expected = [1, 5]
    assert diag(A) == expected, 'Failed test: diag'

    print('Passed all tests.')