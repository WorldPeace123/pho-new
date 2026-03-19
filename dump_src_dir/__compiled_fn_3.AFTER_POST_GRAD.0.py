from __future__ import annotations
import torch
from torch import device
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 1, 1, 1]", primals_2: "f32[3]", primals_3: "f32[1, 1, 3, 3]", primals_4: "f32[3]", primals_5: "f32[3]", primals_6: "f32[3]", primals_7: "f32[3]"):
        # File: /tmp/ipykernel_3208185/1886092087.py:10 in forward, code: x = self.conv(x)
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None

        # File: /tmp/ipykernel_3208185/1886092087.py:11 in forward, code: x = self.bn(x)
        add: "f32[3]" = torch.ops.aten.add.Tensor(primals_5, 1e-05)
        sqrt: "f32[3]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[3]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[3]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
        unsqueeze_1: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        unsqueeze_2: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        sub: "f32[1, 3, 3, 3]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
        mul_1: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_5: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_7: "f32[3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[1, 3, 3, 3]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        return (add_1, primals_1, primals_3, primals_4, primals_5, primals_6, convolution)
