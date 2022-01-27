import typing
import numpy as np
import contraction
import orthogonalize
import primitives

def roundingSumRandThenOrth(tt_tensors1 : typing.List[np.array], tt_tensors2 : typing.List[np.array], desired_ranks : typing.List[int]):
    size = len(tt_tensors1)
    answer = [np.empty(0)] * size
    tt_tensors1_mut = tt_tensors1.copy()
    tt_tensors2_mut = tt_tensors2.copy()
    modes = []
    for tt in tt_tensors1:
        modes.append(tt.shape[1])
    #print(modes)
    random_tensor = primitives.createRandomTensor(modes, desired_ranks)
    contractions1 = contraction.partialContractionsRL(tt_tensors1, random_tensor)
    contractions2 = contraction.partialContractionsRL(tt_tensors2, random_tensor)
    #print(contractions1[0].shape)
    #print(contractions1[1].shape)
    #print(contractions2[0].shape)
    #print(contractions2[1].shape)
    for idx in range(size - 1):
        shape = tt_tensors1_mut[idx].shape
        m1 = primitives.makeVerticalUnfolding(tt_tensors1_mut[idx])
        m2 = primitives.makeVerticalUnfolding(tt_tensors2_mut[idx])
        y = m1 @ contractions1[idx] + m2 @ contractions2[idx]
        y, _ = np.linalg.qr(y)
        answer[idx] = primitives.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m1 = y.T @ m1
        m2 = y.T @ m2
        tt_tensors1_mut[idx + 1] = np.einsum('ij,jkl->ikl', m1, tt_tensors1_mut[idx + 1])
        tt_tensors2_mut[idx + 1] = np.einsum('ij,jkl->ikl', m2, tt_tensors2_mut[idx + 1])
        if idx == size - 2:
            answer[idx + 1] = tt_tensors1_mut[idx + 1] + tt_tensors2_mut[idx + 1]
    return answer