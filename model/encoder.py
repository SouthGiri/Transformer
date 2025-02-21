import copy
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()

        self.layers = []
        for _ in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        
        return out