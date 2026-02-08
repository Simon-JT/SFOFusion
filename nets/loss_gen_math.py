import torch
import torch.nn as nn

class LossGenMath(nn.Module):
    def __init__(self, use_task_grad=True, K=1):  
        super().__init__()
        self.use_task_grad = use_task_grad
        self.K = K
        self.beta  = nn.Parameter(torch.tensor(1e-4))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.eta   = nn.Parameter(torch.tensor(0.5))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def tv_grad(w):
        grad_h = w[:, :, 1:, :] - w[:, :, :-1, :]
        grad_w = w[:, :, :, 1:] - w[:, :, :, :-1]
        denom_h = grad_h.abs() + 1e-6
        denom_w = grad_w.abs() + 1e-6
        tv = torch.zeros_like(w)
        tv[:, :, :-1, :] += grad_h / denom_h
        tv[:, :,  1:, :] -= grad_h / denom_h
        tv[:, :, :, :-1] += grad_w / denom_w
        tv[:, :, :,  1:] -= grad_w / denom_w
        return tv

    @staticmethod
    def norm01(x):
        return x / (x.max() + 1e-6)

    def forward(self, x, meta=False):
        Ia = x[:, :1, :, :]
        Ib = x[:, 1:, :, :]
        If = (Ia + Ib) / 2
        ea = (If - Ia).pow(2)
        eb = (If - Ib).pow(2)
        what = eb / (ea + eb + 1e-6)
        w = what
        for _ in range(self.K):
            tv_g = self.tv_grad(w)
            gradE = (ea - eb) + self.beta * tv_g
            w = torch.clamp(w - self.eta * gradE, 1e-3, 1-1e-3)  
        wa = w
        wb = 1.0 - w
        out = torch.cat([wa, wb], dim=1)
        out = out + 1e-6  
        out = out / out.sum(dim=1, keepdim=True)  
        return out 