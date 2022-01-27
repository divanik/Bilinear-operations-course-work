from mimetypes import init
import typing
import numpy as np

def makeHorizontalUnfolding(tensor : np.array):
    return np.reshape(np.einsum('ijk->ikj', tensor), (tensor.shape[0], -1), order='F')

def fromHorizontalUnfolding(matrix : np.array, shape : typing.Tuple[int]):
    good_shape = (shape[0], shape[2], shape[1])
    return np.einsum('ijk->ikj', np.reshape(matrix, good_shape, order='F'))

def makeVerticalUnfolding(tensor : np.array):
    return np.reshape(tensor, (-1, tensor.shape[2]), order='F')

def fromVerticalUnfolding(matrix : np.array, shape : typing.Tuple[int]):
    return np.reshape(matrix, shape, order='F')

def orthogonalizeRL(tt_tensors : typing.List[np.array]):
    answer = tt_tensors.copy()
    for idx in range(len(tt_tensors) - 1, 0, -1):
        #print(idx)
            #print(answer[idx].shape)
        tensor = answer[idx]
        y = makeHorizontalUnfolding(tensor)
        y, r = np.linalg.qr(y.T)
        initial_shape = (y.shape[1], tensor.shape[1], tensor.shape[2])
        answer[idx] = fromHorizontalUnfolding(y.T, initial_shape)
        #print(answer[idx].shape)
        #print(answer[idx - 1].shape, (r.T).shape)
        answer[idx - 1] = np.einsum('ijk,kl->ijl', answer[idx - 1], r.T)
    return answer
