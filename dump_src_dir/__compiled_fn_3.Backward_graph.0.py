from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 1, 1, 1]", primals_3: "f32[1, 1, 3, 3]", primals_4: "f32[3]", primals_5: "f32[3]", primals_6: "f32[3]", convolution: "f32[1, 3, 3, 3]", tangents_1: "f32[1, 3, 3, 3]"):
        # File: /tmp/ipykernel_3208185/1886092087.py:11 in forward, code: x = self.bn(x)
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 2, 3])
        unsqueeze_8: "f32[1, 3]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        unsqueeze_9: "f32[1, 3, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f32[1, 3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        sub_1: "f32[1, 3, 3, 3]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_10);  convolution = unsqueeze_10 = None
        mul_3: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(tangents_1, sub_1);  sub_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3]);  mul_3 = None
        add: "f32[3]" = torch.ops.aten.add.Tensor(primals_5, 1e-05);  primals_5 = None
        rsqrt: "f32[3]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_8: "f32[3]" = torch.ops.aten.mul.Tensor(rsqrt, primals_6);  primals_6 = None
        unsqueeze_17: "f32[1, 3]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_18: "f32[1, 3, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None
        unsqueeze_19: "f32[1, 3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 3);  unsqueeze_18 = None
        mul_9: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(tangents_1, unsqueeze_19);  tangents_1 = unsqueeze_19 = None
        mul_10: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, rsqrt);  sum_2 = rsqrt = None

        # File: /tmp/ipykernel_3208185/1886092087.py:10 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_9, primals_3, primals_1, [3], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  mul_9 = primals_3 = primals_1 = None
        getitem_1: "f32[3, 1, 1, 1]" = convolution_backward[1]
        getitem_2: "f32[3]" = convolution_backward[2];  convolution_backward = None
        return (getitem_1, getitem_2, None, None, None, mul_10, sum_1)
