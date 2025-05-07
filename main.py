import torch
import torch.nn as nn
import Encoder as enc
import Decoder as dec

# main params:
# vocabSize : size of vocabulary table(suppose ipt and opt use the same vocabulary table)
#    seqLen : max sequence length of input
#  dimModel : feature demension processed in model
#     nHead : head number in multi-head self-attention
# ffnHidden : hidden layer dimension of feed-forward network
#    padIdx : index of <PAD> in vocab table(if change, "padding_idx" param in TokenEmbedding.py should also be changed)
class Transformer(nn.Module):
    def __init__(self, vocabSize, seqLen, dimModel, nHead, ffnHidden, drop, nLayer, device, padIdx = 1):
        super(Transformer, self).__init__()
        self.encoder = enc.Encoder(vocabSize, seqLen, dimModel, nHead, ffnHidden, drop, nLayer, device)
        self.decoder = dec.Decoder(vocabSize, seqLen, dimModel, nHead, ffnHidden, drop, nLayer, device)
        self.softmax = nn.Softmax(dim = -1)
        self.padIdx = padIdx
        self.device = device
    
    # mask of Q @ K.transpose in Encoder MHA and Enc-Dec MHA
    def genPaddingMask(self, xEnc, xDec):
        # size of score matrix is seqLen*seqLen, x is input before embedding
        xLen = xEnc.size(1)
        mEnc = xEnc == self.padIdx
        mDec = xDec == self.padIdx
        # expand rows
        mEnc = mEnc.unsqueeze(1).unsqueeze(2).repeat(1, 1, xLen, 1)
        # expand columns
        mDec = mDec.unsqueeze(1).unsqueeze(3).repeat(1, 1, 1, xLen)
        mask = mDec | mEnc
        return mask
    
    def genCausalMask(self, x):
        # x is decoder input before embedding (decoder input need padding to seqLen)
        xLen = x.size(1)
        mask = torch.triu(torch.ones(xLen, xLen), diagonal = 1).to(torch.bool).to(self.device)
        return mask
    
    def forward(self, xEnc, xDec):
        encMask = self.genPaddingMask(xEnc, xEnc)
        decMask = self.genPaddingMask(xDec, xDec) | self.genCausalMask(xDec, xDec)
        outEnc = self.encoder(xEnc, encMask)
        outDec = self.decoder(outEnc, xDec, decMask, encMask)
        out = self.softmax(outDec)
        return out