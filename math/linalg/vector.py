'''
Vector utilities.
'''

from math import sqrt, atan, pi

def dot(u, v):
    '''Dot product of vectors `u` and `v`'''
    if len(u) != len(v):
        raise ValueError('u and v should have same length')
    return sum(x*y for x,y in zip(u,v))

def norm(v):
    '''Euclidean norm of a vector `v`'''
    return sqrt(sum(x*x for x in v))

def angle(u, v):
    '''Positive angle between 2d vectors `u` and `v` in the plane'''
    if len(u) != 2 or len(v) != 2:
        raise ValueError('u and v should have length 2')
    return abs(atan(v[1]-u[1]) - atan(v[0]-u[0]))

def cross(u, v):
    '''Cross product of (3d) vectors `u` and `v`'''
    if len(u) != 3 or len(v) != 3:
        raise ValueError('u and v should both have length 3')
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ]

def add(u, v):
    '''Sum of `u` and `v`'''
    if len(u) != len(v):
        raise ValueError('expected u and v to have same length')
    return [x+y for x,y in zip(u,v)]

def subtract(u, v):
    '''Difference of `u` and `v`'''
    if len(u) != len(v):
        raise ValueError('expected u and v to have same length')
    return [x-y for x,y in zip(u,v)]

def add_scalar(u, s):
    '''Add vector `u` with scalar `s`'''
    return [x+s for x in u]

def subtract_scalar(u, s):
    '''Subtract scalar `s` from vector `u`'''
    return [x-s for x in u]

def multiply_scalar(u, s):
    '''Multiply vector `u` with scalar `s`'''
    return [x*s for x in u]

def divide_scalar(u, s):
    '''Divide vector `u` by scalar `s`'''
    return [x/s for x in u]

if __name__ == '__main__':
    # test dot
    u = [2, 7, 1]
    v = [8, 2, 8]
    expected = 38
    assert dot(u, v) == expected, 'Failed test: dot'

    # test norm
    v = [2, 3]
    expected = sqrt(13)
    assert norm(v) == expected, 'Failed test: norm'

    # test angle
    u = [0, 1]
    v = [1, 0]
    expected = pi/2
    assert angle(u, v) == expected, 'Failed test: angle'

    # test cross
    u = [-9, -1, 3]
    v = [3, -2, -7]
    expected = [13, -54, 21]
    assert cross(u, v) == expected, 'Failed test: cross'

    # test add
    u = [1, 2, 3]
    v = [4, 5, 6]
    expected = [5, 7, 9]
    assert add(u, v) == expected, 'Failed test: add'

    # test subtract
    u = [1, 2, 3]
    v = [4, 5, 6]
    expected = [-3, -3, -3]
    assert subtract(u, v) == expected, 'Failed test: subtract'

    # test add_scalar
    u = [1, 2, 3]
    s = 4
    expected = [5, 6, 7]
    assert add_scalar(u, s) == expected, 'Failed test: add_scalar'

    # test subtract_scalar
    u = [1, 2, 3]
    s = 4
    expected = [-3, -2, -1]
    assert subtract_scalar(u, s) == expected, 'Failed test: subtract_scalar' 

    # test multiply_scalar
    u = [1, 2, 3]
    s = 4
    expected = [4, 8, 12]
    assert multiply_scalar(u, s) == expected, 'Failed test: multiply_scalar'

    # test divide_scalar
    u = [1, 2, 3]
    s = 4
    expected = [1/4, 1/2, 3/4]
    assert divide_scalar(u, s) == expected, 'Failed test: divide_scalar'

    print('Passed all tests.')