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

class SwitchSequential(nn.Sequential):
     def forward(self, x:torch.Tensor, context: torch.Tensor, time:torch.Tensor) -> torch.Tensor:
          for layer in self:
               if isinstance(layer, UNET_AttentionBlock):
                    x = layer(x, context)
               elif isinstance(layer, UNET_ResidualBlock):
                    x = layer(x, time)
               else:
                    x = layer(x)

          return x
     
class UNET(nn.Module):
     def __init__(self):
          super().__init__()

          self.encoders = nn.Module([
               SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

               SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

               SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

               SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

               SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

               SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

               SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

               SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

               SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

               SwitchSequential(UNET_ResidualBlock(1280, 1280)),

               SwitchSequential(UNET_ResidualBlock(1280, 1280)),
          ])

          self.bottleneck = SwitchSequential(
               UNET_ResidualBlock(1280, 1280),

               UNET_AttentionBlock(8, 160),

               UNET_ResidualBlock(1280, 1280)
          )

          self.decoders = nn.ModuleList([
               SwitchSequential(UNET_ResidualBlock(2560, 1280)),

               SwitchSequential(UNET_ResidualBlock(2560, 1280)),

               SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

               SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_Attention(8, 160)),

               SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_Attention(8, 160)),

               SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_Attention(8, 160), UpSample(1280)),

               SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_Attention(8, 80)),

               SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_Attention(8,80)),

               SwitchSequential(UNET_ResidualBlock(960, 640), UNET_Attention(8, 80), UpSample(640)),

               SwitchSequential(UNET_ResidualBlock(960, 320), UNET_Attention(8, 40)),

               SwitchSequential(UNET_ResidualBlock(640, 320), UNET_Attention(8, 80)),

               SwitchSequential(UNET_ResidualBlock(640, 320), UNET_Attention(8, 40)),
          ])