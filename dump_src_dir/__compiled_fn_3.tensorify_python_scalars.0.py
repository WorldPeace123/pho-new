from __future__ import annotations
import torch
from torch import device
class inner_f(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[3, 1, 1, 1]"; primals_2: "f32[3]"; primals_3: "f32[1, 1, 3, 3]"; primals_4: "f32[3]"; primals_5: "f32[3]"; primals_6: "f32[3]"; primals_7: "f32[3]"; tangents_1: "f32[1, 3, 3, 3]"; 

        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # File: /tmp/ipykernel_3208185/1886092087.py:10 in forward, code: x = self.conv(x)
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(primals_3, primals_1, primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_2 = None

        # File: /tmp/ipykernel_3208185/1886092087.py:11 in forward, code: x = self.bn(x)
        convert_element_type: "f32[3]" = torch.ops.prims.convert_element_type.default(primals_4, torch.float32)
        convert_element_type_1: "f32[3]" = torch.ops.prims.convert_element_type.default(primals_5, torch.float32)
        add: "f32[3]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[3]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[3]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[3]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze: "f32[3, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
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
        add_2: "f32[3]" = torch.ops.aten.add.Tensor(primals_5, 1e-05);  primals_5 = None
        rsqrt: "f32[3]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        unsqueeze_8: "f32[1, 3]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        unsqueeze_9: "f32[1, 3, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        unsqueeze_10: "f32[1, 3, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 2, 3])
        sub_1: "f32[1, 3, 3, 3]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_10);  convolution = unsqueeze_10 = None
        mul_3: "f32[1, 3, 3, 3]" = torch.ops.aten.mul.Tensor(tangents_1, sub_1);  sub_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3]);  mul_3 = None
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
        return pytree.tree_unflatten([add_1, getitem_1, getitem_2, None, None, None, mul_10, sum_1], self._out_spec)
