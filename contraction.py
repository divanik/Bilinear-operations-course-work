import typing
import numpy as np

def PartialContractionsRL(tt_tensors1 : typing.List[np.array], tt_tensors2 : typing.List[np.array]):
    answer = []
    last = np.ones((1, 1))
    for idx, tt1, tt2 in (zip(reversed(range(len(tt_tensors1))), reversed(tt_tensors1), reversed(tt_tensors2))):
        if idx > 0:
            last = np.einsum('ijk,kl,mjl->im', tt1, last, tt2)
            answer.append(last)
    return list(reversed(answer))

#tt1 = [np.array([[3.0, 4.0, ]])]