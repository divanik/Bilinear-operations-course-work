import numpy as np
from src.ttrandombilinear.primitives import countTensor, frob
import src.ttrandombilinear.rounding as rounding


t0 = np.array([ [[4 , 5, 6], [-4, 7, 8], [1, -4, 5]] ])

t1 = np.array([[[2, 3, 4], [5, 2, 8]], [[1, -4, 1], [1, -2, 0]], [[6, -4, -3], [4, 2, -7]]])

t2 = np.array([ [[-2] ,[9], [-1]], [[9], [-10], [1]], [[1], [4], [2]] ])

tt = [t0, t1, t2]

#-------------------------------------------------------

#print(tt2[0].shape)
#print(tt2[1].shape)
#print(tt2[2].shape)

def testOrthogonalizeThenRandomize():
    def testTrivialRounging():
        tt2 = rounding.orthogonalizeThenRandomize(tt, (3, 3))
        
    print(frob(countTensor(tt) - countTensor(tt2)) / np.sqrt(frob(countTensor(tt)) * frob(countTensor(tt2))) )

#-------------------------------------------------------

tt3 = rounding.orthogonalizeThenRandomize(tt,  [3, 3])

#print(tt3[0].shape)
#print(tt3[1].shape)
#print(tt3[2].shape)

#print(countTensor(tt))

#print(countTensor(tt3))

print( frob(countTensor(tt) - countTensor(tt3)) / np.sqrt(frob(countTensor(tt)) * frob(countTensor(tt3))) )

#--------------------------------------------------------

tt4 = rounding.randomizeThenOrthogonalize(tt, [3, 3])

#print(tt4[0].shape)
#print(tt4[1].shape)
#print(tt4[2].shape)

#print(countTensor(tt))

#print(countTensor(tt4))

print( frob(countTensor(tt) - countTensor(tt4)) / np.sqrt(frob(countTensor(tt)) * frob(countTensor(tt4))) )