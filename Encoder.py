import torch
import MHAttention as attn
import LayerNorm as norm
import TokenEmbedding as tkembd
import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, dimModel, dimHidden, drop = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.FC1 = nn.Linear(dimModel, dimHidden)
        self.FC2 = nn.Linear(dimHidden, dimModel)
        self.Dropout = nn.Dropout(p = drop)
    
    def forward(self, x):
        x = self.FC1(x)
        x = x.relu()
        x = self.Dropout(x)
        x = self.FC2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dimModel, ffnHidden, nHead, drop = 0.1):
        super(EncoderLayer, self).__init__()
        self.MHA = attn.MHAttention(dimModel, nHead)
        self.Dropout1 = nn.Dropout(p = drop)
        self.LN1 = norm.LayerNorm(dimModel)
        self.FFN = FeedForwardLayer(dimModel, ffnHidden)
        self.Dropout2 = nn.Dropout(p = drop)
        self.LN2 = norm.LayerNorm(dimModel)

    def forward(self, x, padMask):
        orgX = x
        x = self.MHA(x, x, padMask)
        x = self.Dropout1(x)
        x = self.LN1(x + orgX)
        orgX = x
        x = self.FFN(x)
        x = self.Dropout2(x)
        x = self.LN2(x + orgX)
        return x

class Encoder(nn.Module):
    def __init__(self, vocabSize, seqLen, dimModel, nHead, ffnHidden, drop, nLayer, device):
        super(Encoder, self).__init__()
        self.encEmbed = tkembd.TransformerEmbedding(vocabSize, dimModel, seqLen, drop, device)
        self.encLayer = nn.ModuleList([
            EncoderLayer(dimModel, ffnHidden, nHead, drop) for _ in range(nLayer)
        ])
    
    def forward(self, xEnc, padMask):
        x = self.encEmbed(xEnc)
        for layer in self.encLayer:
            x = layer(x, padMask)
        return x

# test
""" x = torch.tensor([[33, 21, 26, 15], [8, 9, 17, 1]])
model = Encoder(40, 4, 16, 4, 32, 0.1, 2, "cpu")
out = model(x, padMask = None)
print(out) """