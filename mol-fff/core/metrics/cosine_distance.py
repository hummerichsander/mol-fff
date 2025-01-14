import torch.nn as nn
import torch.nn.functional as F


class CosineDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        cosine_similarity = F.cosine_similarity(input1, input2)
        return (1 - cosine_similarity).mean(dim=-1)
