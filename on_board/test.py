import torch
import numpy as np

import bpu_infer_lib

inf = bpu_infer_lib.Infer(False)
inf.load_model("/root/rwkv_v5/rwkv_v5_submodel1.bin")

input_index = 0

instate_0 = torch.rand((1024))
instate_0 = instate_0.numpy().astype(np.float32)
ret = inf.read_numpy_arr_float32(instate_0, input_index)
input_index += 1

input0 = np.array([1]).astype(np.int32)
ret = inf.read_numpy_arr_int32(input0, input_index)
input_index += 1

wkv0 = torch.rand((16,64,64))
wkv0 = wkv0.numpy().astype(np.float32)
ret = inf.read_numpy_arr_float32(wkv0, input_index)
input_index += 1

for i in range(12):
    instate_0 = torch.rand((1024))
    instate_0 = instate_0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(instate_0, input_index)
    input_index += 1
    
    instate_0 = torch.rand((1024))
    instate_0 = instate_0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(instate_0, input_index)
    input_index += 1

    wkv0 = torch.rand((16,64,64))
    wkv0 = wkv0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(wkv0, input_index)
    input_index += 1

print("before infer")
inf.forward(True)
print("finish infer")


input_index = 0

instate_0 = torch.rand((1024))
instate_0 = instate_0.numpy().astype(np.float32)
ret = inf.read_numpy_arr_float32(instate_0, input_index)
input_index += 1

input0 = np.array([1]).astype(np.int32)
ret = inf.read_numpy_arr_int32(input0, input_index)
input_index += 1

wkv0 = torch.rand((16,64,64))
wkv0 = wkv0.numpy().astype(np.float32)
ret = inf.read_numpy_arr_float32(wkv0, input_index)
input_index += 1

for i in range(12):
    instate_0 = torch.rand((1024))
    instate_0 = instate_0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(instate_0, input_index)
    input_index += 1
    
    instate_0 = torch.rand((1024))
    instate_0 = instate_0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(instate_0, input_index)
    input_index += 1

    wkv0 = torch.rand((16,64,64))
    wkv0 = wkv0.numpy().astype(np.float32)
    ret = inf.read_numpy_arr_float32(wkv0, input_index)
    input_index += 1

print("before infer")
inf.forward(False)
print("finish infer")

# #input = np.random.rand(1024).astype(np.float32)
# #print(input.shape)
# #print(input.dtype)
# #state_in = np.random.rand(120, 1024).astype(np.float32)
# #print(state_in.shape)
# #print(state_in.dtype)

# #state_in.tofile("state_in.bin")
# #ret = inf.read_input_float32("state_in.bin", 120*1024, 0)

# #print("ready to infer")
# ret = inf.read_numpy_arr_float32(input, 1)
# ret = inf.read_numpy_arr_float32(state_in, 0)

# print("before infer")
# inf.forward()
# print("finish infer")

# res = inf.get_infer_res_np_float32(0, 50277)
