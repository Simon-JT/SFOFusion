import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#===============================================================================
# FGA Module Implementation
#===============================================================================

class FGA(nn.Module):
    """Frequency Gain Adaptation Module"""
    def __init__(self, n_band=6):
        super().__init__()
        self.n_band = n_band
        self.g_base = nn.Parameter(torch.zeros(n_band))   
        self.mlp = nn.Sequential(
            nn.Linear(n_band, n_band//2),
            nn.GELU(),
            nn.Linear(n_band//2, n_band),
            nn.Sigmoid()
        )
        
    def make_radial_idx(self, amp_log):
        H, W = amp_log.shape[-2:]
        
        center_h, center_w = H // 2, W // 2
        y_coords = torch.arange(H, device=amp_log.device).float()
        x_coords = torch.arange(W, device=amp_log.device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        max_radius = min(center_h, center_w)
        distances = torch.sqrt((y_grid - center_h)**2 + (x_grid - center_w)**2)
        distances = distances / max_radius
        log_distances = torch.log1p(distances * (np.e - 1))  
        band_indices = (log_distances * (self.n_band - 1)).long()
        band_indices = torch.clamp(band_indices, 0, self.n_band - 1)
        return band_indices
    
    def forward(self, amp):  
        amp_log = torch.log1p(amp)
        
        idx = self.make_radial_idx(amp_log)           
        one_hot = F.one_hot(idx, num_classes=self.n_band).permute(2,0,1).float()
        
        e = (amp_log * one_hot).view(amp.shape[0], -1, amp.shape[-2], amp.shape[-1]).mean([2,3])
        
        α = self.mlp(e)
        γ_band = 1. + F.softplus(self.g_base) * α     
        γ_map = γ_band[:, idx]                        
        γ_map = γ_map.unsqueeze(1)                    
        
        return amp * γ_map

#===============================================================================
# Frequency Domain Components
#===============================================================================


class LFSM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.freq_analyzer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.dynamic_kernel_generator = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 8 * (3*3 + 5*5))  
        )

        self.size_weight_mlp = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        self.freq_enhancer = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.freq_adapt = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1), 
            nn.Sigmoid()
        )
        
        self.learnable_threshold = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = nn.Parameter(torch.tensor(5.0))
        self.freq_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 1),
            nn.Sigmoid()
        )
        
    def get_freq_components(self, x, freq_analyzer, threshold_net):
        eps = 1e-7
        original_size = x.shape[-2:]

        fft = torch.fft.rfftn(x, dim=(-2, -1))
        magnitude = torch.abs(fft) + eps
        phase = torch.angle(fft)

        if magnitude.dim() == 3:
            magnitude = magnitude.unsqueeze(1)
        if phase.dim() == 3:
            phase = phase.unsqueeze(1)

        freq_features = freq_analyzer(x)  
        freq_features = freq_features.view(freq_features.size(0), -1) 
        
        kernels_all = self.dynamic_kernel_generator(freq_features)  
        k3_elems = 8 * 3 * 3
        kernels_3 = kernels_all[:, :k3_elems].view(-1, 8, 3, 3)    
        kernels_5 = kernels_all[:, k3_elems:].view(-1, 8, 5, 5)    

        B, C, H, W = magnitude.shape
        filtered_features = []
        size_weights = self.size_weight_mlp(freq_features)  
        
        for i in range(B):
            batch_mag = magnitude[i:i+1]               
            k3 = kernels_3[i]                           
            k5 = kernels_5[i]                            
            f3 = F.conv2d(batch_mag, k3.unsqueeze(1), padding=1)
            f5 = F.conv2d(batch_mag, k5.unsqueeze(1), padding=2)
            w3, w5 = size_weights[i, 0], size_weights[i, 1]
            fused = w3 * f3 + w5 * f5                   
            filtered_features.append(fused)

        filtered_features = torch.cat(filtered_features, dim=0)         
        enhanced_features = self.freq_enhancer(filtered_features)        
        fusion_weights = self.freq_fusion(enhanced_features)  
        fused_features = enhanced_features * fusion_weights  
        filtered = self.freq_adapt(fused_features)  
        freq_mask = torch.sigmoid(
            (filtered - self.learnable_threshold) * self.scale_factor
        )
        
        high_freq = filtered * freq_mask
        low_freq = filtered * (1 - freq_mask)
        
        return {
            'high_freq': high_freq.clamp(0, 5.0),
            'low_freq': low_freq.clamp(0, 5.0),
            'phase': phase,
            'spatial_size': original_size,
            'freq_size': magnitude.shape[-2:]
        }
        
    def forward(self, vis_img, ir_img):
        rgb_components = self.get_freq_components(
            vis_img, self.freq_analyzer, self.freq_adapt
        )
        
        ir_components = self.get_freq_components(
            ir_img, self.freq_analyzer, self.freq_adapt
        )
        
        return {
            'rgb': {
                'low_freq': rgb_components['low_freq'],
                'high_freq': rgb_components['high_freq'],
                'phase': rgb_components['phase'],
                'spatial_size': rgb_components['spatial_size'],
                'freq_size': rgb_components['freq_size']
            },
            'ir': {
                'low_freq': ir_components['low_freq'],
                'high_freq': ir_components['high_freq'],
                'phase': ir_components['phase'],
                'spatial_size': ir_components['spatial_size'],
                'freq_size': ir_components['freq_size']
            }
        }



#===============================================================================
# Multi-Spectral Attention Components
#===============================================================================

class MultiSpectralAttentionLayer(nn.Module):

    def __init__(self, channel, reduction=4,
                 freq_sel=[(0,0),(0,1),(1,0),(1,1)]):
        super().__init__()
        self.freq_sel = freq_sel
        hidden = max(channel // reduction, 1)

        self.fc1 = nn.Conv1d(channel,      hidden, 1, bias=False)
        self.fc2 = nn.Conv1d(hidden, channel, 1, bias=False)

    def _multi_spectral_pool(self, x):
        B, C, H, W = x.shape
        
        try:
            dct = torch.fft.dct(torch.fft.dct(x, dim=-1, norm='ortho'),
                                dim=-2, norm='ortho')
        except AttributeError:
            dct = self._custom_dct2d(x)
        
        coeffs = []
        for (u,v) in self.freq_sel:
            coeffs.append(dct[..., u, v])              
        return torch.stack(coeffs, dim=-1)              
    
    def _custom_dct2d(self, x):
        B, C, H, W = x.shape
        
        def dct1d(x, dim):
            N = x.shape[dim]
            n = torch.arange(N, device=x.device, dtype=x.dtype).view(-1, 1)
            k = torch.arange(N, device=x.device, dtype=x.dtype).view(1, -1)
            
            import math
            cos_term = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
            dct_matrix = cos_term * torch.sqrt(torch.tensor(2.0 / N, device=x.device, dtype=x.dtype))
            dct_matrix[0] *= torch.sqrt(torch.tensor(0.5, device=x.device, dtype=x.dtype))  # 第一行特殊处理
            
            if dim == -1:
                return torch.matmul(x, dct_matrix.t())
            else:
                return torch.matmul(dct_matrix, x)
        
        x_dct_w = dct1d(x, dim=-1)
        x_dct_hw = dct1d(x_dct_w, dim=-2)
        
        return x_dct_hw

    def forward(self, x):
        B, C, H, W = x.size()
        spectral = self._multi_spectral_pool(x)         
        y = self.fc1(spectral)                          
        y = F.relu(y, inplace=True)
        y = self.fc2(y)                                
        y = torch.sigmoid(y.mean(-1, keepdim=True))    
        y = y.view(B, C, 1, 1)                         
        return x * y                                    


#===============================================================================
# Spatial Domain Edge Enhancement
#===============================================================================

class EdgeLearnable(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.highpass_gain = nn.Parameter(torch.tensor(0.4))
        self.edge_conv = nn.Conv2d(1, 2, 3, 1, 1, bias=False)
        
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        
        with torch.no_grad():
            self.edge_conv.weight[0] = sobel_x
            self.edge_conv.weight[1] = sobel_y
        
        def make_gaussian_kernel(ksize=5, sigma=1.0):
            ax = torch.arange(ksize) - (ksize - 1) / 2.0
            xx, yy = torch.meshgrid(ax, ax)
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel
        gauss5 = make_gaussian_kernel(5, 1.0).view(1, 1, 5, 5)
        self.register_buffer('gaussian_kernel', gauss5)
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )
    
    def forward(self, x):

        edges = self.edge_conv(x)  
        edge_mag = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2 + 1e-6)      
        low = F.conv2d(x, self.gaussian_kernel, padding=2)
        high = x - low
        beta = torch.sigmoid(self.highpass_gain) 
        x_hp = x + beta * high
        combined = torch.cat([edge_mag, edges[:, 0:1], edges[:, 1:2], x_hp], dim=1)  
        edge_features = self.fuse(combined)
        return x + edge_features


class DifferenceSignificanceAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.significance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.local_significance = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, diff_map):
        diff_feat = self.diff_encoder(diff_map)        
        global_weight = self.significance_head(diff_feat)  
        diff_feat_weighted = diff_feat * global_weight  
        significance_mask = self.local_significance(diff_feat_weighted)  
        
        return significance_mask


class CompensationGate(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()  
        )
        
        self.compensation_scale = nn.Parameter(torch.tensor(0.5)) 
        
    def forward(self, significance_mask, freq_output):
        combined = torch.cat([significance_mask, freq_output], dim=1)  # (B, 2, H, W)
        base_strength = self.gate_network(combined)  # (B, 1, H, W)
        scale = torch.sigmoid(self.compensation_scale)
        compensation_strength = base_strength * scale
        
        return compensation_strength


class GuidedSpatialExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.compensation_generator = nn.Sequential(
            nn.Conv2d(8 + 1, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, ir_enhanced, vis_enhanced, compensation_strength):
        spatial_input = torch.cat([ir_enhanced, vis_enhanced], dim=1)  
        spatial_features = self.spatial_encoder(spatial_input)  
        compensation_input = torch.cat([spatial_features, compensation_strength], dim=1)  
        compensation_features = self.compensation_generator(compensation_input)  
        compensation_features = compensation_features * compensation_strength 
        
        return compensation_features


class DifferenceGuidedAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.local_attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, diff_map):
        diff_feat = self.diff_encoder(diff_map)      
        global_attn = self.global_attention(diff_feat)  
        diff_feat_global = diff_feat * global_attn  
        local_attn = self.local_attention(diff_feat_global)  
        combined_attn = torch.cat([local_attn, diff_map], dim=1)  
        attention_weights = self.attention_fusion(combined_attn)  
        
        return attention_weights


class ResidualCompensation(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.coarse_compensation = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.fine_compensation = nn.Sequential(
            nn.Conv2d(8 + 1, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.residual_gate = nn.Sequential(
            nn.Conv2d(8 + 1, 8, kernel_size=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, ir_enhanced, vis_enhanced, attention_weights):
        spatial_input = torch.cat([ir_enhanced, vis_enhanced], dim=1)  # (B, 2, H, W)
        coarse_features = self.coarse_compensation(spatial_input)  # (B, 8, H, W)
        fine_input = torch.cat([coarse_features, attention_weights], dim=1)  # (B, 9, H, W)
        fine_features = self.fine_compensation(fine_input)  # (B, 8, H, W)
        gate_input = torch.cat([fine_features, attention_weights], dim=1)  # (B, 9, H, W)
        residual_gate = self.residual_gate(gate_input)  # (B, 1, H, W)
        residual_features = fine_features * residual_gate  # (B, 8, H, W)
        
        return residual_features


class AmpFuse(nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),  
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, amp_vis, amp_ir):          
        combined = torch.cat([amp_vis, amp_ir], dim=1)  
        alpha = self.gate_network(combined)              
        amp_f = alpha * amp_vis + (1 - alpha) * amp_ir
        return amp_f

class PhaFuse(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.d_model = d_model
        
        self.vis_q = nn.Conv2d(1, d_model, 1)
        self.ir_k = nn.Conv2d(1, d_model, 1)
        self.ir_v = nn.Conv2d(1, d_model, 1)
        
        self.out_proj = nn.Conv2d(d_model, 1, 1)
        
    def forward(self, pha_vis, pha_ir):
        B, _, H, W = pha_vis.shape
        
        Q = self.vis_q(pha_vis)  
        K = self.ir_k(pha_ir)    
        V = self.ir_v(pha_ir)    

        attention_map = torch.sigmoid(torch.sum(Q * K, dim=1, keepdim=True))  
        attended_ir = attention_map * V  
        combined = Q + attended_ir  
        pha_f = self.out_proj(combined)  
        pha_f = torch.remainder(pha_f + math.pi, 2 * math.pi) - math.pi
        
        return pha_f

class PhaseAlign(nn.Module):
    def __init__(self, in_ch=1, hid=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hid, 1, 3, 1, 1),   
            nn.Tanh()
        )

    def forward(self, vis_phi, ir_phi):
        delta = ir_phi - vis_phi                       
        M = self.net(delta)                            
        ir_phi_corr = ir_phi - M * delta                
        return ir_phi_corr, delta, M

class PhaseGuidedAmpFuse(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.phase_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),  
            nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU(),  
            nn.Conv2d(8, 4, 1)                    
        )
        
        self.fusion_net = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1), nn.ReLU(), 
            nn.Conv2d(16, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 1), nn.Sigmoid()
        )

    def forward(self, A_vis, A_ir, phase_diff):
        phase_features = self.phase_feature_extractor(phase_diff)  
        combined = torch.cat([A_vis, A_ir, phase_features], dim=1) 
        alpha = self.fusion_net(combined) 
        
        fused_amp = alpha * A_vis + (1 - alpha) * A_ir
        return fused_amp, alpha

class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        attention = self.conv(avg_map)
        return x * attention

class DDFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase_align = PhaseAlign()
        self.phase_fuse = PhaFuse()
        self.amp_fuse_phase_guided = PhaseGuidedAmpFuse()
        self.amp_fuse = AmpFuse()
        self.edge_extractor = EdgeLearnable()
        self.freq_light_gate = MultiSpectralAttentionLayer(channel=1, reduction=4)
        self.alpha_gate = nn.Parameter(torch.tensor(0.3))  
        self.diff_analyzer_ir = DifferenceSignificanceAnalyzer()
        self.diff_analyzer_vis = DifferenceSignificanceAnalyzer()
        self.compensation_gate_ir = CompensationGate()
        self.compensation_gate_vis = CompensationGate()
        self.guided_spatial_extractor = GuidedSpatialExtractor()
        self.freq_align = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )
        self.spatial_align = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(8)
        )
        self.spatial_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=1),
            nn.Sigmoid()
        )
        self.interaction_dw = nn.Conv2d(9, 9, kernel_size=3, padding=1, groups=9, bias=False)
        self.interaction_pw = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 9, kernel_size=1, bias=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_freq_info, ir_freq_info, spatial_ir, spatial_vis):
        vis_amplitude = rgb_freq_info['low_freq'] + rgb_freq_info['high_freq']
        ir_amplitude = ir_freq_info['low_freq'] + ir_freq_info['high_freq']
        rgb_phase = rgb_freq_info['phase']
        ir_phase = ir_freq_info['phase']
        
        ir_phase_corr, delta_phi, M = self.phase_align(rgb_phase, ir_phase)
        
        fused_phase = self.phase_fuse(rgb_phase, ir_phase_corr)
 
        fused_amplitude, alpha_map = self.amp_fuse_phase_guided(
            vis_amplitude, ir_amplitude, delta_phi.abs())
        
        real = fused_amplitude * torch.cos(fused_phase)
        imag = fused_amplitude * torch.sin(fused_phase)
        freq_complex = torch.complex(real.squeeze(1), imag.squeeze(1))
        
        spatial_size = rgb_freq_info['spatial_size']
        
        freq_output = torch.fft.irfftn(
            freq_complex,
            s=spatial_size,
            dim=(-2, -1)
        ).unsqueeze(1)
        
        alpha = torch.sigmoid(self.alpha_gate)
        gated = self.freq_light_gate(freq_output) * freq_output
        freq_output = (1 - alpha) * freq_output + alpha * gated
        
        diff_vis = torch.abs(spatial_vis - freq_output)  
        diff_ir = torch.abs(spatial_ir - freq_output)    
        
        significance_vis = self.diff_analyzer_vis(diff_vis) 
        significance_ir = self.diff_analyzer_ir(diff_ir)     
        compensation_strength_vis = self.compensation_gate_vis(significance_vis, freq_output)  
        compensation_strength_ir = self.compensation_gate_ir(significance_ir, freq_output)       
        compensation_strength = torch.max(compensation_strength_vis, compensation_strength_ir)  
        ir_enhanced = self.edge_extractor(spatial_ir)  
        vis_enhanced = self.edge_extractor(spatial_vis)  
        spatial_features = self.guided_spatial_extractor(
            ir_enhanced, vis_enhanced, compensation_strength
        )  
        freq_aligned = self.freq_align(freq_output)
        spatial_aligned = self.spatial_align(spatial_features)
        spatial_aligned = spatial_aligned * self.spatial_se(spatial_aligned)
        combined_features = torch.cat([freq_aligned, spatial_aligned], dim=1) 

        dw = self.interaction_dw(combined_features)
        logits = self.interaction_pw(dw)                          
        interaction_weights = 0.2 + 0.6 * torch.sigmoid(logits)   
        fused_features = combined_features * interaction_weights
        output = self.final_conv(fused_features)
        
        output = (output - output.min()) / (output.max() - output.min() + 1e-6)
        
        fused_freq_info = {
            'amp': fused_amplitude,
            'phase': fused_phase
        }
        
        return output, fused_freq_info


class SFOFusion(nn.Module):
    def __init__(self, n_band=6):
        super().__init__()
        self.freq_separator = LFSM()
        self.freq_enhancement = FGA(n_band=n_band)
        self.final_fusion = DDFM()
        
    def forward(self, vis_img, ir_img):
        if len(vis_img.shape) == 3:
            vis_img = vis_img.unsqueeze(1)
        if len(ir_img.shape) == 3:
            ir_img = ir_img.unsqueeze(1)
        freq_components = self.freq_separator(vis_img, ir_img)
        rgb_full_amp = freq_components['rgb']['low_freq'] + freq_components['rgb']['high_freq']
        ir_full_amp = freq_components['ir']['low_freq'] + freq_components['ir']['high_freq']
        rgb_amp_enhanced = self.freq_enhancement(rgb_full_amp)
        ir_amp_enhanced = self.freq_enhancement(ir_full_amp)
        rgb_amp_ratio = freq_components['rgb']['high_freq'] / (rgb_full_amp + 1e-6)
        ir_amp_ratio = freq_components['ir']['high_freq'] / (ir_full_amp + 1e-6)
        freq_components['rgb']['high_freq'] = rgb_amp_enhanced * rgb_amp_ratio
        freq_components['rgb']['low_freq'] = rgb_amp_enhanced * (1 - rgb_amp_ratio)
        freq_components['ir']['high_freq'] = ir_amp_enhanced * ir_amp_ratio
        freq_components['ir']['low_freq'] = ir_amp_enhanced * (1 - ir_amp_ratio)
        output, freq_info = self.final_fusion(
            freq_components['rgb'],
            freq_components['ir'],
            ir_img,  
            vis_img  
        )
        
        return output, freq_info



