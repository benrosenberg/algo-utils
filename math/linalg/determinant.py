'''
Simple functions for 2- and 3-dimensional matrix determinants.

For \\(n\\times n\\) determinants, see the function `PLUDeterminant` 
in `decomposition`.
'''

def det2(A):
    '''Determinant of a 2 by 2 matrix `A`'''
    if len(A) != 2 or len(A[0]) != 2:
        raise ValueError('matrix must have dimensions (2,2)')
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

def det3(A):
    '''Determinant of a 3 by 3 matrix `A`'''
    if len(A) != 3 or len(A[0]) != 3:
        raise ValueError('matrix must have dimensions (3,3)')
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    return a*(e*i - f*h) - b*(d*i - g*f) + c*(d*h - e*g)

if __name__ == '__main__':
    # test det2
    A = [
        [-1, -3],
        [3, 1]
    ]
    expected = 8
    assert det2(A) == expected, 'Failed test: det2'

    # test det3
    A = [
        [1, 3, 2],
        [-3, -1, -3],
        [2, 3, 1]
    ]
    expected = -15
    assert det3(A) == expected, 'Failed test: det3'

    print('Passed all tests.')