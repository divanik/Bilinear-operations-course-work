import typing
import numpy as np

def partialContractionsRL(tt_tensors1 : typing.List[np.array], tt_tensors2 : typing.List[np.array]):
    answer = []
    last = np.ones((1, 1))
    for idx, tt1, tt2 in (zip(reversed(range(len(tt_tensors1))), reversed(tt_tensors1), reversed(tt_tensors2))):
        if idx > 0:
            last = np.einsum('ijk,kl,mjl->im', tt1, last, tt2)
            answer.append(last)
    return list(reversed(answer))

def partialContractionsLR(tt_tensors1 : typing.List[np.array], tt_tensors2 : typing.List[np.array]):
    answer = []
    cur = np.ones((1, 1))
    for idx, tt1, tt2 in zip(range(len(tt_tensors1)), tt_tensors1, tt_tensors2):
        if idx < (len(tt_tensors1) - 1):
            cur = np.einsum('ijk,il,ljm->km', tt1, cur, tt2)
            answer.append(cur)
    return answer

#tt1 = [np.array([[3.0, 4.0, ]])]