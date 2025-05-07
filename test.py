import torch
import torch.nn as nn

""" posEcd = torch.zeros(8, 6, dtype = float)
pos = torch.arange(0, 8, dtype = float)
pos = pos.unsqueeze(dim = 1)
_2i = torch.arange(0, 6, step = 2, dtype = float)
posEcd[:, 0::2] = pos * torch.sin(_2i)
posEcd[:, 1::2] = pos * torch.cos(_2i)
print(posEcd) """

""" x = torch.tensor([[[1, 2, 3], [2, 3, 1]], [[3, 4, 2], [2, 1, 4]]])
y = torch.tensor([[[3, 2], [2, 3], [1, 2]], [[4, 2], [1, 4], [2, 3]]])
out = x @ y
print(out)
mask = torch.tensor([[[False, True], [False, False]]])
out.masked_fill_(mask, 0)
print(out) """

""" x = torch.tensor([[2, 3, 1], [3, 1, 1]])
x = x == 1
x = x.unsqueeze(2).repeat(1, 1, 3)
x = x | x.transpose(1, 2)
print(x) """

""" x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = torch.tensor([False, False, True]).unsqueeze(0)
x = x.masked_fill(mask, 0)
print(x) """

x = torch.tensor([[1, 2], [4, 5]])
x = x.ne(2).unsqueeze(1).unsqueeze(3)
print(x.size())
x = x.repeat(1, 1, 1, 2)
print(x)
print(x.size())