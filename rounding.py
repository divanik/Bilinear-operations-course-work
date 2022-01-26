import typing
import numpy as np
import orthogonalize

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
        answer[idx] = orthogonalize.fromVerticalUnfolding(y, shape)

        print()

        print(tensor.shape)
        print(y.shape)
        print(r.shape)
        U, S, Vt = np.linalg.svd(r)
        l = desired_ranks[idx]
        U = U[:, :l]
        S = S[:l]
        Vt = Vt[:l, :]
        print(answer[idx].shape, U.shape)
        answer[idx] = np.einsum('ijk,kl->ijl', answer[idx], U)
        print(S.shape, Vt.shape)
        SVt = np.einsum('i,ij->ij', S, Vt)
        print(SVt.shape, answer[idx + 1].shape)
        answer[idx + 1] = np.einsum('ij,jkl->ikl', SVt, answer[idx + 1])
    return answer

'''
def orthogonalizeThenRandomize(tt_tensors : typing.List[np.array], desired_ranks : typing.List[int]):
    answer = orthogonalizeRL(tt_tensors)
    for idx in range(len(answer) - 1):
        tensor = answer[idx]
        shape = tensor.shape
        z = makeVerticalUnfolding(tensor)
        omega = np.random.normal(loc=0.0, scale = 1.0 / (z.shape[1] * desired_ranks[idx]), size = (z.shape[1], desired_ranks[idx]))
        y = z @ omega
        v, _ = np.linalg.qr(y)
        #m = v.T @ 
        answer[idx] = fromVerticalUnfolding(y, shape)
        U, S, Vt = np.linalg.svd(r)
        l = desired_ranks[idx]
        U = U[:, :l]
        S = S[:l]
        Vt = Vt[:l, :]
        answer[idx] = np.einsum('ijk,kl->ijl', answer[idx], U)
        SVt = np.einsum('i,ij->ij', S, Vt)
        answer[idx + 1] = np.einsum('ij,jkl->ikl', SVt, answer[idx + 1])
    return answer
'''