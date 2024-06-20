'''
Utilities for determining the connected components of a graph.
'''

def find(parent, a):
    '''Find the root ancestor of `a` in `parent`'''
    while a != parent[a]:
        a = parent[a]
    return a

def union(parent, size, a, b):
    '''Combine components `a` and `b` in `parent` and `size`
    
    Input:
    
     - `parent`: map `i` \\(\\mapsto\\) `parent[i]`
     - `size`: map `i` \\(\\mapsto\\) `size[i]`
     - `a`: graph node
     - `b`: graph node
     
    Output:

     - `parent'`: updated `parent` array
     - `size'`: updated `size` array
    '''
    a_parent = find(parent, a)
    b_parent = find(parent, b)
    if a_parent == b_parent: return parent, size
    size_a = size[a_parent]
    size_b = size[b_parent]
    if size_a > size_b:
        parent[b_parent] = a_parent
        size[a_parent] += size[b_parent]
    else:
        parent[a_parent] = b_parent
        size[b_parent] += size[a_parent]
    return parent, size

def connected(parent, a, b):
    '''Whether `a` and `b` share an ancestor in `parent`'''
    return find(parent, a) == find(parent, b)

def connected_components(edges):
    '''Connected components of the graph specified by `edges`
    
    Input:
    
     - `edges`: a list of 2-tuples (or 2-iterables)
     
    Output:
    
     - `components`: a dictionary {`ancestor`:`children`} of lists of components that share ancestors

    Uses the Union-Find algorithm.
    '''
    parent = {}
    size = {}
    for i,j in edges:
        parent[i] = i
        parent[j] = j
        size[i] = 1
        size[j] = 1
    for i,j in edges:
        parent, size = union(parent, size, i, j)
    out = {i:[i] for i in parent if parent[i] == i}
    for i in parent:
        if i != parent[i]:
            out[find(parent, i)].append(i)
    return out
    
if __name__ == '__main__':
    # test union_find
    edges = [
        [1, 1],
        [3, 4],
        [2, 3]
    ]
    components = connected_components(edges)
    unambiguous = [sorted(component) for component in components.values()]
    assert unambiguous == [[1], [2, 3, 4]], 'Failed test: connected_components'