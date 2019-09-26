import torch
import numpy as np

a = torch.Tensor([[7, 2], [3, 4], [8, 6]])
mask = a[..., 1]
print(mask)
mask = a[..., 1] > 4
print(mask)

idxs = mask.nonzero()
print(idxs)
yecs = a[mask]
print(yecs)

print(idxs[:,0])
print(yecs[:, 1:])



