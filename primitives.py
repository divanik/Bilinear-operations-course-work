import typing
import numpy as np

def countTensor(tt_tensors : typing.List[np.array]):
    answer = np.ones(1)
    for tensor in tt_tensors:
        answer = np.einsum('...i,ijk->...jk', answer, tensor)
    return np.einsum('...i,i->...', answer, np.ones(1))

def frob(tensor : np.array):
    return np.sqrt(np.sum(tensor * tensor))