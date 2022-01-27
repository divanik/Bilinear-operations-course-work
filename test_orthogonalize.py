import numpy as np
import orthogonalize
import primitives

t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

#t2 = np.array([ [[-2] ,[9]], [[9], [-10]], [[1], [4]] ])
t2 = np.array([ [[-2] ,[9], [0]], [[9], [-10], [3]], [[1], [4], [9]] ])

'''
print(t0.shape)
print(t1.shape)
print(t2.shape)

hor0 = orthogonalize.makeHorizontalUnfolding(t0)
print(hor0)

ver0 = orthogonalize.makeVerticalUnfolding(t0)
print(ver0)

hor1 = orthogonalize.makeHorizontalUnfolding(t1)
print(hor1)

ver1 = orthogonalize.makeVerticalUnfolding(t1)
print(ver1)

hor2 = orthogonalize.makeHorizontalUnfolding(t2)
print(hor2)

ver2 = orthogonalize.makeVerticalUnfolding(t2)
print(ver2)
'''

tt = [t0, t1, t2]

tt2 = orthogonalize.orthogonalizeRL(tt)

'''
print(tt[0].shape)
print(tt[1].shape)
print(tt[2].shape)

print(tt2[0].shape)
print(tt2[1].shape)
print(tt2[2].shape)
'''

U2 = orthogonalize.makeHorizontalUnfolding(tt2[2])
U1 = orthogonalize.makeHorizontalUnfolding(tt2[1])

#print(np.eye(U2.shape[0]) - U2 @ U2.T)

assert( np.linalg.norm(np.eye(U2.shape[0]) - U2 @ U2.T, ord='fro') < 1e-10)

assert( np.linalg.norm(np.eye(U1.shape[0]) - U1 @ U1.T, ord='fro') < 1e-10)

assert(primitives.frob(primitives.countTensor(tt) - primitives.countTensor(tt2)) < 1e-10)