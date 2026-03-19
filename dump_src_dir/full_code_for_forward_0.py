
# Note: the following variables are used inside the guard function.
___check_tensors = '''None'''
___check_tensors_verbose = '''None'''
___check_global_state = '''<built-in method check of torch._C._dynamo.guards.GlobalStateGuard object at 0x14765424b4f0>'''
___check_torch_function_mode_stack = '''<function make_torch_function_mode_stack_guard.<locals>.check_torch_function_mode_stack at 0x147658e1be20>'''
IsNonOverlappingAndDenseIndicator = '''<function eval_is_non_overlapping_and_dense at 0x147671c56a20>'''
cast_symbool_to_symint_guardless = '''<function cast_symbool_to_symint_guardless at 0x147671c56c00>'''
math = '''<module 'math' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/lib-dynload/math.cpython-313-x86_64-linux-gnu.so'>'''
torch = '''<module 'torch' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/__init__.py'>'''
___check_type_id = '''<built-in function check_type_id>'''
___check_obj_id = '''<built-in function check_obj_id>'''
___odict_getitem = '''<method '__getitem__' of 'dict' objects>'''
___key_to_id = '''<function key_to_id at 0x147669e12de0>'''
___dict_version = '''<built-in function dict_version>'''
___dict_contains = '''<function _get_closure_vars.<locals>.<lambda> at 0x147655921940>'''
___tuple_iterator_len = '''<method '__length_hint__' of 'tuple_iterator' objects>'''
___normalize_range_iter = '''<function normalize_range_iter at 0x147669e12700>'''
___tuple_iterator_getitem = '''<function tuple_iterator_getitem at 0x147669e125c0>'''
___dataclass_fields = '''<function dataclass_fields at 0x147669e12660>'''
___namedtuple_fields = '''<function _get_closure_vars.<locals>.<lambda> at 0x1476557fa3e0>'''
___get_torch_function_mode_stack_at = '''<function get_torch_function_mode_stack_at at 0x147669e1aac0>'''
___get_current_stream = '''<function get_current_stream at 0x147669e1aca0>'''
__math_isnan = '''<built-in function isnan>'''
__numpy_isnan = '''<ufunc 'isnan'>'''
inf = '''inf'''
__load_module = '''<function import_module at 0x1477d90954e0>'''
utils_device = '''<module 'torch.utils._device' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/utils/_device.py'>'''
device = '''<class 'torch.device'>'''
___from_numpy = '''<function from_numpy at 0x14766997fa60>'''
___as_tensor = '''<function _as_tensor_fullprec at 0x147673222d40>'''
inspect = '''<module 'inspect' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/inspect.py'>'''
def guard_0(L, G, **___kwargs_ignored):
    __guard_hit = True
    __guard_hit = __guard_hit and torch._functorch.aot_autograd.utils.top_saved_tensors_hooks ids == None  # _dynamo/output_graph.py:866 in init_ambient_guards
    __guard_hit = __guard_hit and ___check_global_state() against {"allow_bf16_reduce":0,"allow_fp16_reduce":0,"allow_tf32":false,"autocast_state":{"cached_enabled":true,"dtype":[15,5,5,15,5,5,15,15,5,5],"enabled":[false,false,false,false,false,false,false,false,false,false]},"default_dtype":6,"deterministic_algorithms":false,"deterministic_algorithms_warn_only":false,"grad_mode":true,"num_threads":32,"torch_function":true,"torch_function_all_disabled":false}
    __guard_hit = __guard_hit and ___check_torch_function_mode_stack()
    __guard_hit = __guard_hit and utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:853 in init_ambient_guards
    __guard_hit = __guard_hit and check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[1, 1, 3, 3], stride=[9, 9, 3, 1])  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and hasattr(L['x'], '_dynamo_dynamic_indices') == False           # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_conv'], 22498997589136), type=<module 'torch.nn.modules.conv' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/nn/modules/conv.py'>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_conv'].F, 22499002237792), type=<module 'torch.nn.functional' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/nn/functional.py'>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_conv'].F.conv2d, 22504893371376), type=<built-in method conv2d of type object at 0x1477c38161a0>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_conv'].F.batch_norm.__code__, 94211360898384), type=<code object batch_norm at 0x55af4ab33d50, file "/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/nn/functional.py", line 2811>  # return F.batch_norm(  # nn/modules/batchnorm.py:194 in forward
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_module'], 22499005066480), type=<module 'torch.nn.modules.module' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/nn/modules/module.py'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward
    __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 94211240250784), type=<class 'collections.OrderedDict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type OrderedDict)
    __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 94211240250784), type=<class 'collections.OrderedDict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type OrderedDict)
    __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 94211240250784), type=<class 'collections.OrderedDict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type OrderedDict)
    __guard_hit = __guard_hit and ___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 94211240250784), type=<class 'collections.OrderedDict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type OrderedDict)
    __guard_hit = __guard_hit and ___check_obj_id(G['__import_torch_dot_nn_dot_modules_dot_batchnorm'], 22498997467664), type=<module 'torch.nn.modules.batchnorm' from '/opt/local/miniconda3/envs/py3md/lib/python3.13/site-packages/torch/nn/modules/batchnorm.py'>  # return F.batch_norm(  # nn/modules/batchnorm.py:194 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self'], 94211371834560), type=<class '__main__.MyModule'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type MyModule)
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules, 94211240258976), type=<class 'dict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type dict)
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['bn'], 94211362144880), type=<class 'torch.nn.modules.batchnorm.BatchNorm2d'>  # x = self.bn(x)  # mp/ipykernel_3208185/1886092087.py:11 in forward (HINT: type BatchNorm2d)
    __guard_hit = __guard_hit and not ___dict_contains('forward', L['self']._modules['bn'].__dict__)  # x = self.bn(x)  # mp/ipykernel_3208185/1886092087.py:11 in forward
    __guard_hit = __guard_hit and not ___dict_contains('_check_input_dim', L['self']._modules['bn'].__dict__)  # self._check_input_dim(input)  # nn/modules/batchnorm.py:161 in forward
    __guard_hit = __guard_hit and L['self']._modules['bn'].eps == 1e-05                         # return F.batch_norm(  # nn/modules/batchnorm.py:194 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['bn']._buffers, 94211240258976), type=<class 'dict'>  # bn_training = (self.running_mean is None) and (self.running_var is None)  # nn/modules/batchnorm.py:187 in forward (HINT: type dict)
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['bn']._buffers['running_var'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # self.running_var if not self.training or self.track_running_stats else None,  # nn/modules/batchnorm.py:202 in forward
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['bn']._buffers['running_mean'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[3], stride=[1])  # bn_training = (self.running_mean is None) and (self.running_var is None)  # nn/modules/batchnorm.py:187 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['bn']._modules, 94211240258976), type=<class 'dict'>  # bn_training = (self.running_mean is None) and (self.running_var is None)  # nn/modules/batchnorm.py:187 in forward (HINT: type dict)
    __guard_hit = __guard_hit and L['self']._modules['bn'].momentum == 0.1                      # if self.momentum is None:  # nn/modules/batchnorm.py:166 in forward
    __guard_hit = __guard_hit and L['self']._modules['bn'].training == False                    # if self.training and self.track_running_stats:  # nn/modules/batchnorm.py:171 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['bn']._parameters, 94211240258976), type=<class 'dict'>  # bn_training = (self.running_mean is None) and (self.running_var is None)  # nn/modules/batchnorm.py:187 in forward (HINT: type dict)
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['bn']._parameters['bias'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[3], stride=[1])  # self.bias,  # nn/modules/batchnorm.py:204 in forward
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['bn']._parameters['weight'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[3], stride=[1])  # self.weight,  # nn/modules/batchnorm.py:203 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv'], 94211362983568), type=<class 'torch.nn.modules.conv.Conv2d'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type Conv2d)
    __guard_hit = __guard_hit and not ___dict_contains('forward', L['self']._modules['conv'].__dict__)  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward
    __guard_hit = __guard_hit and not ___dict_contains('_conv_forward', L['self']._modules['conv'].__dict__)  # return self._conv_forward(input, self.weight, self.bias)  # nn/modules/conv.py:553 in forward
    __guard_hit = __guard_hit and L['self']._modules['conv'].groups == 1                        # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward (HINT: torch.compile considers integer attributes of the nn.Module to be static. If you are observing recompilation, you might want to make this integer dynamic using torch._dynamo.config.allow_unspec_int_on_nn_module = True, or convert this integer into a tensor.)
    __guard_hit = __guard_hit and L['self']._modules['conv'].stride == (1, 1)                   # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv'].stride, 94211240227648), type=<class 'tuple'>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward (HINT: type tuple)
    __guard_hit = __guard_hit and len(L['self']._modules['conv'].stride) == 2                   # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and L['self']._modules['conv'].padding == (0, 0)                  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv'].padding, 94211240227648), type=<class 'tuple'>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward (HINT: type tuple)
    __guard_hit = __guard_hit and len(L['self']._modules['conv'].padding) == 2                  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and L['self']._modules['conv'].dilation == (1, 1)                 # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv'].dilation, 94211240227648), type=<class 'tuple'>  # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward (HINT: type tuple)
    __guard_hit = __guard_hit and len(L['self']._modules['conv'].dilation) == 2                 # return F.conv2d(  # nn/modules/conv.py:548 in _conv_forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv']._parameters, 94211240258976), type=<class 'dict'>  # return self._conv_forward(input, self.weight, self.bias)  # nn/modules/conv.py:553 in forward (HINT: type dict)
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['conv']._parameters['bias'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[3], stride=[1])  # return self._conv_forward(input, self.weight, self.bias)  # nn/modules/conv.py:553 in forward
    __guard_hit = __guard_hit and check_tensor(L['self']._modules['conv']._parameters['weight'], Parameter, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=True, size=[3, 1, 1, 1], stride=[1, 1, 1, 1])  # return self._conv_forward(input, self.weight, self.bias)  # nn/modules/conv.py:553 in forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._modules['conv'].padding_mode, 94211240209952), type=<class 'str'>  # if self.padding_mode != "zeros":  # nn/modules/conv.py:535 in _conv_forward (HINT: type str)
    __guard_hit = __guard_hit and L['self']._modules['conv'].padding_mode == 'zeros'            # if self.padding_mode != "zeros":  # nn/modules/conv.py:535 in _conv_forward
    __guard_hit = __guard_hit and ___check_type_id(L['self']._parameters, 94211240258976), type=<class 'dict'>  # x = self.conv(x)  # mp/ipykernel_3208185/1886092087.py:10 in forward (HINT: type dict)
    __guard_hit = __guard_hit and G['__import_torch_dot_nn_dot_modules_dot_conv'].F is G['__import_torch_dot_nn_dot_modules_dot_batchnorm'].F  # return F.batch_norm(  # nn/modules/batchnorm.py:194 in forward
    return __guard_hit

# Note: please refer to the graph code in __compiled_fn_3_db829f5e_ac59_4c3d_a2b0_75dbc0360a4b*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3_db829f5e_ac59_4c3d_a2b0_75dbc0360a4b(*args, **kwargs):
    pass

def transformed_code_0(self, x):
    'Failed to decompile.'


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x,

def transformed_forward(self, x):
    __local_dict = {"self": self, "x": x}
    __global_dict = globals()
    if guard_0(__local_dict, __global_dict):
        return transformed_code_0(self, x)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, x)

#============ end of forward ============#
