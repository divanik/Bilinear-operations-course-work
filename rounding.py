import typing
import numpy as np
import orthogonalize
import contraction

def ttRoundingWithRanks(tt_tensors : typing.List[np.array], desired_ranks : typing.List[int]):
    answer = orthogonalize.orthogonalizeRL(tt_tensors)
    print(answer[0].shape)
    print(answer[1].shape)
    print(answer[2].shape)
    for idx in range(len(answer) - 1):
        tensor = answer[idx]
        shape = tensor.shape
        y = orthogonalize.makeVerticalUnfolding(tensor)
        y, r = np.linalg.qr(y)
        shape = (tensor.shape[0], tensor.shape[1], y.shape[1])
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, shape)

        #print(tensor.shape)
        #print(y.shape)
        #print(r.shape)
        print(answer[idx].shape)
        print(r.shape)
        U, S, Vt = np.linalg.svd(r, full_matrices=False)
        l = desired_ranks[idx]
        print(l, U.shape, S.shape, Vt.shape)
        if l < S.size:
            U = U[:, :l]
            S = S[:l]
            Vt = Vt[:l, :]
        #print(answer[idx].shape, U.shape)
        answer[idx] = np.einsum('ijk,kl->ijl', answer[idx], U)
        print(l, U.shape, S.shape, Vt.shape)

        SVt = np.diag(S) @ Vt
        #print(SVt.shape, answer[idx + 1].shape)
        answer[idx + 1] = np.einsum('ij,jkl->ikl', SVt, answer[idx + 1])
    return answer

def orthogonalizeThenRandomize(tt_tensors : typing.List[np.array], desired_ranks : typing.List[int]):
    answer = orthogonalize.orthogonalizeRL(tt_tensors)
    for idx in range(len(answer) - 1):
        tensor = answer[idx]
        z = orthogonalize.makeVerticalUnfolding(tensor)
        omega = np.random.normal(loc=0.0, scale = 1/(z.shape[1] *  desired_ranks[idx]), size = (z.shape[1], desired_ranks[idx]))  #there is a question about scale
        print(f'Omega: {omega}')
        y = z @ omega
        v, _ = np.linalg.qr(y)
        m = v.T @ z
        answer[idx + 1] = np.einsum('ij,jkl->jkl', m, answer[idx + 1])
    return answer

def createRandomTensor(modes : typing.List[np.array], ranks : typing.List[np.array]):
    answer = []
    for idx in len(modes):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(loc = 0.0, scale = 1 / (l1 * modes[idx] * l2), size = (l1, modes[idx], l2))
        answer.append(tensor)
    return tensor

def randomizeThenOrthogonalize(tt_tensors : typing.List[np.array], desired_ranks : typing.List[int]):
    modes = []
    for tt in tt_tensors:
        modes.append(tt.shape[1])
    random_tensor = createRandomTensor(modes, desired_ranks)
    contractions = contraction.partialContractionsRL(tt_tensors, random_tensor)
    answer = tt_tensors.copy()
    for idx in len(tt_tensors) - 1:
        z = orthogonalize.makeVerticalUnfolding(answer[idx])
        shape = answer[idx].shape
        y = z * contractions[idx]
        y, _ = np.linalg.qr(y)
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m = y.T @ z
        answer[idx + 1] = np.einsum('ij, jkl->ikl', m, answer[idx + 1])

'''
def twoSidedRandomization(tt_tensors : typing.List[np.array], 
            desired_ranks : typing.List[int], helping_ranks : typing.List[int]):
    modes = []
    for tt in tt_tensors:
        modes.append(tt.shape[1])
    small_random_tensor = createRandomTensor(modes, desired_ranks)
    big_random_tensor = createRandomTensor(modes, helping_ranks)
    left_contractions = contraction.partialContractionsRL(tt_tensors, small_random_tensor)
    right_contractions = contraction.partialContractionsRL(tt_tensors, big_random_tensor)
    for idx in len(tt_tensors) - 1:
        U, S, Vt = np.linalg.svd(left_contractions[idx] @ right_contractions[idx])
        z = orthogonalize.makeVerticalUnfolding(answer[idx])
        shape = answer[idx].shape
        y = z * contractions[idx]
        y, _ = np.linalg.qr(y)
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m = y.T @ z
        answer[idx + 1] = np.einsum('ij, jkl->ikl', m, answer[idx + 1])
'''

def roundingSumRandThenOrth(tt_tensors1 : typing.List[np.array], tt_tensors2 : typing.List[np.array], desired_ranks : typing.List[int]):
    size = len(tt_tensors1)
    answer = [np.empty(0)] * size
    tt_tensors1_mut = tt_tensors1.copy()
    tt_tensors2_mut = tt_tensors2.copy()
    modes = []
    for tt in tt_tensors1:
        modes.append(tt.shape[1])
    random_tensor = createRandomTensor(modes, desired_ranks)
    contractions1 = contraction.partialContractionsRL(tt_tensors1, random_tensor)
    contractions2 = contraction.partialContractionsRL(tt_tensors2, random_tensor)
    for idx in range(size - 1):
        shape = tt_tensors1_mut[idx].shape
        m1 = orthogonalize.makeVerticalUnfolding(tt_tensors1_mut[idx]) 
        m2 = orthogonalize.makeVerticalUnfolding(tt_tensors2_mut[idx])
        y = m1 @ contractions1[idx] + m2 @ contractions2[idx]
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m1 = y.T @ m1
        m2 = y.T @ m2
        tt_tensors1_mut[idx + 1] = np.einsum('ij,jkl', m1, tt_tensors1_mut[idx + 1])
        tt_tensors2_mut[idx + 1] = np.einsum('ij,jkl', m2, tt_tensors2_mut[idx + 1])
        if idx == size - 2:
            answer[idx + 1] = tt_tensors1_mut[idx + 1] + tt_tensors2_mut[idx + 1]
    return answer