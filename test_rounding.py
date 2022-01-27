import numpy as np
import contraction
import orthogonalize
from primitives import countTensor, frob
import rounding


t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([ [[-2] ,[9], [-1]], [[9], [-10], [1]], [[1], [4], [2]] ])

print(t0.shape)
print(t1.shape)
print(t2.shape)

tt = [t0, t1, t2]

tt2 = rounding.ttRoundingWithRanks(tt, (2, 2))

print(tt2[0].shape)
print(tt2[1].shape)
print(tt2[2].shape)

print(frob(countTensor(tt) - countTensor(tt2)) / frob(countTensor(tt)))

tt3 = rounding.orthogonalizeThenRandomize(tt, (3, 3))

print(tt3[0].shape)
print(tt3[1].shape)
print(tt3[2].shape)

print(countTensor(tt))

print(countTensor(tt3))

print(frob(countTensor(tt) - countTensor(tt3)) / frob(countTensor(tt)))
