# AOT ID: ['4_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__native_batch_norm_legit_no_training_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(16L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(9L); x1+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(3L) && x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(9L)))
                    {
                        alignas(std::max(std::size_t(16), alignof(float))) float tmp0[16*16];
                        transpose_mxn<float,static_cast<int64_t>(9L),static_cast<int64_t>(3L),false>(in_ptr0 + static_cast<int64_t>(x0 + 3L*x1), static_cast<int64_t>(3L), tmp0, static_cast<int64_t>(9L));
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(3L); x0_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(9L*x0_inner), static_cast<int64_t>(9L));
                            auto tmp2 = in_ptr1[static_cast<int64_t>(x0 + x0_inner)];
                            auto tmp5 = in_ptr2[static_cast<int64_t>(x0 + x0_inner)];
                            auto tmp15 = in_ptr3[static_cast<int64_t>(x0 + x0_inner)];
                            auto tmp18 = in_ptr4[static_cast<int64_t>(x0 + x0_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = float(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = static_cast<int32_t>(1);
                            auto tmp10 = tmp9 / tmp8;
                            auto tmp11 = static_cast<float>(1.0);
                            auto tmp12 = float(tmp10 * tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp4 * tmp13;
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp17 + tmp19;
                            tmp20.store(out_ptr0 + static_cast<int64_t>(x1 + 9L*x0 + 9L*x0_inner), static_cast<int64_t>(9L));
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
        args.clear()
        assert_size_stride(primals_1, (3, 1, 1, 1), (1, 1, 1, 1))
        assert_size_stride(primals_2, (3, ), (1, ))
        assert_size_stride(primals_3, (1, 1, 3, 3), (9, 9, 3, 1))
        assert_size_stride(primals_4, (3, ), (1, ))
        assert_size_stride(primals_5, (3, ), (1, ))
        assert_size_stride(primals_6, (3, ), (1, ))
        assert_size_stride(primals_7, (3, ), (1, ))
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1, 1, 3, 3), (9, 1, 3, 1), 0), primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
        assert_size_stride(buf0, (1, 3, 3, 3), (27, 1, 9, 3), 'torch.ops.aten.convolution.default')
        del primals_2
        buf1 = empty_strided_cpu((1, 3, 3, 3), (27, 9, 3, 1), torch.float32)
        cpp_fused__native_batch_norm_legit_no_training_0(buf0, primals_4, primals_5, primals_6, primals_7, buf1)
        del primals_7
        return (buf1, primals_1, primals_3, primals_4, primals_5, primals_6, buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
