import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss_gen_math import LossGenMath   

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)

class LightAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class LossGenLite(nn.Module):
    def __init__(self, use_task_grad=True, C=12):
        super().__init__()
        self.use_task_grad = use_task_grad
        self.C = C

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, C, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(C),
            
            nn.Conv2d(C, C, 3, 1, 1, groups=C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(C*2),
            
            nn.Conv2d(C*2, C, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(C),
        )
        
        self.attention_blocks = nn.Sequential(
            LightAttentionBlock(C),
            LightAttentionBlock(C),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(C, C//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C//2, 1, 1)
        )

        self.beta  = nn.Parameter(torch.tensor(1e-4))  
        self.eta   = nn.Parameter(torch.tensor(0.4))   
        self.lam   = nn.Parameter(torch.tensor(0.5))   
        self.sigma = nn.Parameter(torch.tensor(1.0))   

    def correntropy(self, diff):
        sigma = torch.clamp(self.sigma, 0.1, 10.)
        return 1 - torch.exp(-diff.pow(2) / (2*sigma**2))

    def forward(self, x, task_grad=None):
        Ia, Ib = x[:, :1], x[:, 1:]

        features = self.feature_extractor(x)
        
        enhanced_features = self.attention_blocks(features)
        
        w_log = self.fusion(enhanced_features)
        w = torch.sigmoid(w_log)

        If = (Ia + Ib) / 2                      
        diff_a, diff_b = If - Ia, If - Ib

        Ea = self.lam * diff_a.pow(2) + (1-self.lam) * self.correntropy(diff_a)
        Eb = self.lam * diff_b.pow(2) + (1-self.lam) * self.correntropy(diff_b)

        grad = Ea - Eb + self.beta * LossGenMath.tv_grad(w)
        if self.use_task_grad and task_grad is not None:
            grad = grad + task_grad

        w = torch.sigmoid(w_log - self.eta * grad)
        wa, wb = w, 1 - w
        out = torch.cat([wa, wb], 1) + 1e-6
        out = out / out.sum(1, keepdim=True)
        return out



