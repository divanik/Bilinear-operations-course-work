import typing
import numpy as np

def countTensor(tt_tensors : typing.List[np.array]):
    answer = np.ones(1)
    for tensor in tt_tensors:
        answer = np.einsum('...i,ijk->...jk', answer, tensor)
    return np.einsum('...i,i->...', answer, np.ones(1))

def frob(tensor : np.array):
    return np.sqrt(np.sum(tensor * tensor))

def createRandomTensor(modes : typing.List[np.array], ranks : typing.List[np.array]):
    answer = []
    #print('?-----?')
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(loc = 0.0, scale = 1 / (l1 * modes[idx] * l2), size = (l1, modes[idx], l2))
        #print(tensor.shape)
        answer.append(tensor)
    return answer

def makeHorizontalUnfolding(tensor : np.array):
    return np.reshape(np.einsum('ijk->ikj', tensor), (tensor.shape[0], -1), order='F')

def fromHorizontalUnfolding(matrix : np.array, shape : typing.Tuple[int]):
    good_shape = (shape[0], shape[2], shape[1])
    return np.einsum('ijk->ikj', np.reshape(matrix, good_shape, order='F'))

def makeVerticalUnfolding(tensor : np.array):
    return np.reshape(tensor, (-1, tensor.shape[2]), order='F')

def fromVerticalUnfolding(matrix : np.array, shape : typing.Tuple[int]):
    return np.reshape(matrix, shape, order='F')