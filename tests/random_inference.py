import torch
import einops

from ..models import *

if __name__ == "__main__":

    config = TransformerConfig()
    model = Transformer(config)
    input = einops.rearrange(torch.arange(8), "(b l) -> b l", b = 2)

    for _ in range(4):
        output = torch.argmax(model(input)[:, -1], dim=-1) # to get the argmax token of the last position
        print(output)
        input = torch.cat([input, output.unsqueeze(1)], dim=-1)

    print(input)
