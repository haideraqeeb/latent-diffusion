import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class Diffusion(nn.Module):
     def __init__(self):
          self.time_embedding = TimeEmbedding(320)
          self.unet = UNET()
          self.final = UNET_OutputLayer(320, 4)

     def forward(self, latent: torch.Tensor, context: torch.Tensor, time:torch.Tensor):
          time = self.time_embedding(time)
          output = self.unet(latent, context, time)
          output = self.final(output)

          return output
     
class TimeEmbedding(nn.Module):
     def __init__(self, n_embd: int):
          super().__init__()
          self.linear1 = nn.Linear(n_embd, 4*n_embd)
          self.linear2 = nn.Linear(4*n_embd, 4*n_embd)

     def forward(self, x: torch.Tensor) -> torch.Tensor:
          x = self.linear1(x)
          x = F.silu(x)
          x = self.linear2(x)

          return x
