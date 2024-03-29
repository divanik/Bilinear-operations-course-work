import numpy as np
from bilinear_package.src import orthogonalize
from bilinear_package.src import primitives

t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

#t2 = np.array([ [[-2] ,[9]], [[9], [-10]], [[1], [4]] ])
t2 = np.array([ [[-2] ,[9], [0]], [[9], [-10], [3]], [[1], [4], [9]] ])


tt = [t0, t1, t2]

tt2 = orthogonalize.orthogonalizeRL(tt)


print(tt[0].shape)
print(tt[1].shape)
print(tt[2].shape)

'''
print(tt2[0].shape)
print(tt2[1].shape)
print(tt2[2].shape)
'''

U2 = primitives.makeHorizontalUnfolding(tt2[2])
U1 = primitives.makeHorizontalUnfolding(tt2[1])

#print(np.eye(U2.shape[0]) - U2 @ U2.T)


def test_orthoganality():
    assert( np.linalg.norm(np.eye(U2.shape[0]) - U2 @ U2.T, ord='fro') < 1e-10)
    assert( np.linalg.norm(np.eye(U1.shape[0]) - U1 @ U1.T, ord='fro') < 1e-10)

def test_correctness():
    assert(primitives.frob(primitives.countTensor(tt) - primitives.countTensor(tt2)) < 1e-10)

