import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    # vocabSize: size of vocabulary table, dimModel: dimension of a token after embedding
    # padding token will be filled in "0" after embedding
    def __init__(self, vocabSize, dimModel):
        super(TokenEmbedding, self).__init__(vocabSize, dimModel, padding_idx = 1)

class PosEmbedding(nn.Module):
    def __init__(self, dimModel, seqLen, device):
        # seqLen: superparam, xLen(token num of x) < seqLen -> padding to seqLen, xLen > seqLen -> truncate
        super(PosEmbedding, self).__init__()
        # 2 dimension zero matrix, size = [seqLen, dimModel]
        self.posEcd = torch.zeros(seqLen, dimModel, device = device)
        self.posEcd.requires_grad = False
        pos = torch.arange(0, seqLen, device = device)
        pos = pos.unsqueeze(dim = 1)
        # here, dim of pos: [seqLen] --> [seqLen, 1]
        _2i = torch.arange(0, dimModel, step = 2)
        # so far, dimension of pos is [seqLen, 1], dimension of _2i is [dimModel/2]
        self.posEcd[:, 0::2] = torch.sin(pos / 10000**(_2i / dimModel))
        self.posEcd[:, 1::2] = torch.cos(pos / 10000**(_2i / dimModel))
        # dim of left side: [seqLen, odd/even column number]
        # dim of right side: [row of pos, col of _2i]->[seqLen, dimModel/2]
        # Attention: if dimModel is odd, the 2nd formula report error because left dim is not equal to right
    
    def forward(self, x):
        # input x have 2(3) params: [batchSize, wrdLen(, dimModel)]
        batchSize, wrdLen = x.size()[:2]
        # in this case, input token len(wrdLen) should be less than seqLen
        return self.posEcd[:wrdLen, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocabSize, dimModel, seqLen, pDrop, device):
        super(TransformerEmbedding ,self).__init__()
        # class instantiation: automatically run __init__ function
        self.tokenEmbed = TokenEmbedding(vocabSize, dimModel)
        self.posEmbed = PosEmbedding(dimModel, seqLen, device)
        # dropout can avoid overfitting
        self.DropOut = nn.Dropout(p = pDrop)
    
    def forward(self, x):
        iptTkEmbed = self.tokenEmbed(x)
        iptPosEmbed = self.posEmbed(x)
        return self.DropOut(iptTkEmbed + iptPosEmbed)
    
# test
# elements in tensor x is index of token input
""" x = torch.tensor([[6, 2, 5], [4, 3, 1]])
model = TokenEmbedding(10, 8)
out = model(x)
print(out) """

""" x = torch.tensor([[2, 3, 1], [4, 3, 6]])
model = TransformerEmbedding(10, 16, 4, 0.1, "cpu")
out = model(x)
print(out)
print(out.dtype) """