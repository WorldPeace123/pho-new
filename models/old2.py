#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 2: MDN-Copula Output Distribution Model
对击中PMT的光子建模输出(t, q)的联合分布

模型架构：
- 混合密度网络 (Mixture Density Network)
- t: 指数分布 (Exponential distribution)
- q: 正态分布 (Normal distribution)
- Copula: 高斯Copula建模t和q的相关性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class MDN_Copula(nn.Module):
    """
    Mixture Density Network with Gaussian Copula
    
    输入: 光子特征 (6维)
    输出: 混合模型参数
        - pi: 混合权重
        - t_rates: 时间的指数分布参数
        - q_means, q_stds: 电荷的正态分布参数
        - rho: Copula相关系数
    """
    def __init__(self, input_dim=6, num_components=8):
        super().__init__()
        
        # 共享特征提取网络
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        
        # 各个参数的输出头
        self.pi_head = nn.Linear(256, num_components)
        self.t_rate_head = nn.Linear(256, num_components)
        self.q_mean_head = nn.Linear(256, num_components)
        self.q_std_head = nn.Linear(256, num_components)
        self.rho_head = nn.Linear(256, num_components)

    def forward(self, x):
        """
        前向传播，输出混合模型的所有参数
        
        Args:
            x: (B, 6) 输入特征
            
        Returns:
            pi_logits: (B, K) 混合权重logits
            t_rates: (B, K) 时间指数分布的rate参数
            q_means: (B, K) 电荷正态分布的均值
            q_stds: (B, K) 电荷正态分布的标准差
            rho: (B, K) Copula相关系数
        """
        feats = self.shared_net(x)
        pi_logits = self.pi_head(feats)
        
        eps = 1e-6
        t_rates = F.softplus(self.t_rate_head(feats)) + eps
        q_means = self.q_mean_head(feats)
        q_stds = F.softplus(self.q_std_head(feats)) + eps
        rho = torch.tanh(self.rho_head(feats))
        
        return pi_logits, t_rates, q_means, q_stds, rho

    @torch.no_grad()
    def sample(self, x, max_attempts=50):
        """
        从混合模型中采样 (t, q)
        
        采样流程:
        1. 根据混合权重选择一个分量
        2. 使用Copula从二维高斯采样
        3. 通过逆CDF变换得到 (t, q)
        4. 对q<0的样本进行重采样
        
        Args:
            x: (B, 6) 输入特征
            max_attempts: 最大重采样次数
            
        Returns:
            samples: (B, 2) 采样结果 [t, q]
        """
        pi_logits, t_rates, q_means, q_stds, rho = self.forward(x)

        # 选择混合分量
        mix = D.Categorical(logits=pi_logits)
        comp = mix.sample()
        idx = torch.arange(x.size(0), device=x.device)

        # 获取选中分量的参数
        rate = t_rates[idx, comp]
        mu = q_means[idx, comp]
        sig = q_stds[idx, comp]
        r = rho[idx, comp]

        # 初始化采样结果
        t = torch.zeros(x.size(0), device=x.device)
        q = torch.zeros(x.size(0), device=x.device)
        need = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # 重采样循环（处理q<0的情况）
        for _ in range(max_attempts):
            if not need.any():
                break
            n = int(need.sum().item())

            # 构建协方差矩阵
            cov = torch.zeros(n, 2, 2, device=x.device)
            cov[:, 0, 0] = 1.0
            cov[:, 1, 1] = 1.0
            cov[:, 0, 1] = r[need]
            cov[:, 1, 0] = r[need]

            # 从二维高斯采样
            mvn = D.MultivariateNormal(torch.zeros(n, 2, device=x.device), cov)
            z = mvn.sample()

            # 通过逆CDF变换得到t (指数分布)
            u = D.Normal(0, 1).cdf(z[:, 0])
            u = torch.clamp(u, 0.0, 1.0 - 1e-7)
            t_new = -torch.log(1.0 - u) / rate[need]
            
            # 通过标准正态变换得到q (正态分布)
            q_new = mu[need] + sig[need] * z[:, 1]

            t[need] = t_new
            q[need] = q_new
            
            # 更新需要重采样的mask
            need = (q < 0)

        # 对仍然为负的q设置一个小的正值
        q[need] = 0.01
        
        return torch.stack([t, q], dim=1)