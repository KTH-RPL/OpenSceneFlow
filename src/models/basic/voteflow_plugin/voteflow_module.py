import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class ConvBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size, stride, padding, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        return x

class ConvBNBlock(nn.Module):
    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm2d(out_num_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
class VolConvBN(nn.Module):
    def __init__(self, h, w, hidden_dim=16, dim_output=64):
        super().__init__()
        assert h%2==0
        assert w%2==0
        self.conv1 = ConvBNBlock(in_num_channels=1, out_num_channels=hidden_dim, stride=2)
        self.conv2 = ConvBNBlock(in_num_channels=hidden_dim, out_num_channels=hidden_dim, stride=2)
        self.linear = nn.Linear(math.ceil(h/4) * math.ceil(w/4) * hidden_dim, dim_output)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _, h, w = x.shape
        x = self.conv1(x.view(b*l, 1, h, w))
        x = self.conv2(x)
        # print(x.shape)
        x = self.linear(x.view(b*l, -1))
        # print('after linear:', x.shape)
        x = self.relu(x)
        return x.view(b, l, -1)

    
class VoteFlowLinearDecoder(nn.Module):
    def __init__(self, dim_input=16, layer_size=1, filter_size=128):
        super().__init__()
        # self.linear = nn.Linear(m*dim_input, dim_output)
        offset_encoder_channels = 128
        self.offset_encoder = nn.Linear(3, offset_encoder_channels)
        filter_size=filter_size

        decoder = nn.ModuleList()
        decoder.append(torch.nn.Linear(dim_input + offset_encoder_channels, filter_size))
        decoder.append(torch.nn.GELU())
        for _ in range(layer_size-1):
            decoder.append(torch.nn.Linear(filter_size, filter_size))
            decoder.append(torch.nn.ReLU())
        decoder.append(torch.nn.Linear(filter_size, 3))

        self.decoder = nn.Sequential(*decoder)
        print(self.decoder)
        
    def forward(self, x: torch.Tensor, pts_offsets: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        pts_offsets_feats = self.offset_encoder(pts_offsets)
        x = torch.cat([x, pts_offsets_feats], dim=-1)
        # print('decoder conv: ', x.shape)
        x = self.decoder(x)
        
        return x
