import torch
import torch.nn as nn
import MHAttention as attn
import LayerNorm as norm
import Encoder as ffn
import TokenEmbedding as tkembd

class DecoderLayer(nn.Module):
    def __init__(self, dimModel, nHead, ffnHidden, drop = 0.1):
        super(DecoderLayer, self).__init__()
        self.MMHA = attn.MHAttention(dimModel, nHead)
        self.Dropuot1 = nn.Dropout(p = drop)
        self.LN1 = norm.LayerNorm(dimModel)
        self.CrossMHA = attn.MHAttention(dimModel, nHead)
        self.Dropout2 = nn.Dropout(p = drop)
        self.LN2 = norm.LayerNorm(dimModel)
        self.FFN = ffn.FeedForwardLayer(dimModel, ffnHidden)
        self.Dropout3 = nn.Dropout(p = drop)
        self.LN3 = norm.LayerNorm(dimModel)
    
    def forward(self, xEnc, xDec, futureMask, crossMask):
        orgDec = xDec
        xDec = self.MMHA(xDec, xDec, futureMask)
        xDec = self.Dropuot1(xDec)
        xDec = self.LN1(xDec + orgDec)
        orgDec = xDec
        xDec = self.CrossMHA(xEnc, xDec, crossMask)
        xDec = self.Dropout2(xDec)
        xDec = self.LN2(xDec + orgDec)
        orgDec = xDec
        xDec = self.FFN(xDec)
        xDec = self.Dropout3(xDec)
        xDec = self.LN3(xDec + orgDec)
        return xDec

class Decoder(nn.Module):
    def __init__(self, vocabSize, seqLen, dimModel, nHead, ffnHidden, drop, nLayer, device):
        super(Decoder, self).__init__()
        self.decEmbed = tkembd.TransformerEmbedding(vocabSize, dimModel, seqLen, drop, device)
        self.decLayer = nn.ModuleList([
            DecoderLayer(dimModel, nHead, ffnHidden, drop) for _ in range(nLayer)
        ])
        self.FC = nn.Linear(dimModel, vocabSize)
    
    # params: xDec: Decoder input, xEnc: Encoder output, futureMask: mask of MMHA
    def forward(self, xEnc, xDec, futureMask, crossMask):
        x = self.decEmbed(xDec)
        for layer in self.decLayer:
            x = layer(xEnc, x, futureMask, crossMask)
        x = self.FC(x)
        return x

# test
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
""" xEnc = torch.rand(2, 4, 16)
xDec = torch.tensor([[12, 25, 31, 1], [11, 18, 26, 3]])
mask = torch.zeros(2, 4, 4, 4, dtype = bool)
model = Decoder(40, 4, 16, 4, 32, 0.1, 2, "cpu")
out = model(xEnc, xDec, mask)
print(out) """