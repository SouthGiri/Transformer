import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff

    def forward(self, src, src_mask):
        out = src
        out = self.self_attention(query=src, key=src, value=src, mask=src_mask)
        out = self.position_ff(out)

        return out
        