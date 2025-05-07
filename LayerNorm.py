import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dimModel, epsilon = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dimModel))
        self.beta = nn.Parameter(torch.zeros(dimModel))
        self.epsilon = epsilon
    
    def forward(self, x):
        # dim = -1: the highest index, e.g.size[1, 2, 3(2, also -1)]  size[2, 4, 3, 5(3, also -1)]
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, unbiased = False, keepdim = True)
        # Attention: math.sqrt() only suppport scalar, torch.sqrt() support tensor
        out = (x - mean) / torch.sqrt(var + self.epsilon)
        out = self.gamma * out + self.beta
        return out

# test
""" x = torch.rand(2, 3, 4)
print(x)
model = LayerNorm(4)
out = model(x)
print(out) """