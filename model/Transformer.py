import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out
    
    def decode(self, z, c):
        out = self.decode(z, c)
        return out
    
    def forward(self, src, z, src_mask):
        encoder_out = self.encode(src, src_mask)
        out = self.decode(z, encoder_out)
        return out