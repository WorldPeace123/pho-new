from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_conv_parameters_weight_: "f32[3, 1, 1, 1]", L_self_modules_conv_parameters_bias_: "f32[3]", L_x_: "f32[1, 1, 3, 3]", L_self_modules_bn_buffers_running_mean_: "f32[3]", L_self_modules_bn_buffers_running_var_: "f32[3]", L_self_modules_bn_parameters_weight_: "f32[3]", L_self_modules_bn_parameters_bias_: "f32[3]"):
        l_self_modules_conv_parameters_weight_ = L_self_modules_conv_parameters_weight_
        l_self_modules_conv_parameters_bias_ = L_self_modules_conv_parameters_bias_
        l_x_ = L_x_
        l_self_modules_bn_buffers_running_mean_ = L_self_modules_bn_buffers_running_mean_
        l_self_modules_bn_buffers_running_var_ = L_self_modules_bn_buffers_running_var_
        l_self_modules_bn_parameters_weight_ = L_self_modules_bn_parameters_weight_
        l_self_modules_bn_parameters_bias_ = L_self_modules_bn_parameters_bias_

        # File: /tmp/ipykernel_3208185/1886092087.py:10 in forward, code: x = self.conv(x)
        x: "f32[1, 3, 3, 3]" = torch.conv2d(l_x_, l_self_modules_conv_parameters_weight_, l_self_modules_conv_parameters_bias_, (1, 1), (0, 0), (1, 1), 1);  l_x_ = l_self_modules_conv_parameters_weight_ = l_self_modules_conv_parameters_bias_ = None

        # File: /tmp/ipykernel_3208185/1886092087.py:11 in forward, code: x = self.bn(x)
        x_1: "f32[1, 3, 3, 3]" = torch.nn.functional.batch_norm(x, l_self_modules_bn_buffers_running_mean_, l_self_modules_bn_buffers_running_var_, l_self_modules_bn_parameters_weight_, l_self_modules_bn_parameters_bias_, False, 0.1, 1e-05);  x = l_self_modules_bn_buffers_running_mean_ = l_self_modules_bn_buffers_running_var_ = l_self_modules_bn_parameters_weight_ = l_self_modules_bn_parameters_bias_ = None
        return (x_1,)
