import torch
import math
import torch.nn as nn

class MHAttention(nn.Module):
    def __init__(self, dimModel, nHead):
        super(MHAttention, self).__init__()
        self.nHead = nHead
        self.wQ = nn.Linear(dimModel, dimModel)
        self.wK = nn.Linear(dimModel, dimModel)
        self.wV = nn.Linear(dimModel, dimModel)
        self.softmax = nn.Softmax(dim = -1)
        self.wOut = nn.Linear(dimModel, dimModel)
    
    def forward(self, xEnc, xDec, mask = None):
        # here, dimFeature = dimModel. Intrinsically, these two value is the same
        # x: input after tokenembedding, size is [batchSize, seqLen, dimModel]
        batchSize, seqLen, dimFeature = xEnc.size()
        dimHead = dimFeature // self.nHead
        # Linear layer: Q = X @ wQ
        Q = self.wQ(xDec)
        K = self.wK(xEnc)
        V = self.wV(xEnc)
        # .view() is similiar to .reshape(), .permute(): exchange dimension order of tensor
        Q = Q.view(batchSize, seqLen, self.nHead, dimHead).permute(0, 2, 1, 3)
        K = K.view(batchSize, seqLen, self.nHead, dimHead).permute(0, 2, 1, 3)
        V = V.view(batchSize, seqLen, self.nHead, dimHead).permute(0, 2, 1, 3)
        score = (Q @ K.transpose(2, 3)) / math.sqrt(dimHead)
        if(mask != None):
            # masked_fill(): return new tensor, masked_fill_(): in-palce modification
            score.masked_fill_(mask, -1e5)
        score = self.softmax(score) @ V
        # .contiguous(): ensure tensor is store in memory continuously, use between permute/view or transpose/view
        score = score.permute(0, 2, 1, 3).contiguous().view(batchSize, seqLen, dimFeature)
        out = self.wOut(score)
        return out

# test
""" x = torch.rand(1, 8, 64)
print(x.dtype)
model = MHAttention(64, 8)
out = model(x, x)
print(x.dtype)
print(out) """