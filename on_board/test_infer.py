import os
import numpy as np
import bpu_infer_lib

main_directory = '/root/rwkv_v5/dumped_inputs_submodel1_bak'
file_paths = []

for root, dirs, files in os.walk(main_directory):
    # Iterate through subdirectories
    subdirs = sorted(dirs)
    for subdir in subdirs:
        subdir_path = os.path.join(root, subdir)
        # print(f"Subdirectory: {subdir_path}")
        
        # Iterate through files in the subdirectory
        for subroot, _, subfiles in os.walk(subdir_path):
            sorted_subfiles = sorted(subfiles)
            for file in sorted_subfiles:
                file_path = os.path.join(subroot, file)
                # print(f"File: {file_path}")
                file_paths.append(file_path)
                
print(file_paths)
state_files = file_paths[1:26]
wkv_files = file_paths[26:39]
# print(len(wkv_files))

state = []
for state_file in state_files:
    state.append(np.fromfile(state_file, dtype=np.float32))
print(state[-1].shape)
state2 = []
for wkv in wkv_files:
    state2.append(np.fromfile(wkv, dtype=np.float32))
    print(state2[-1].shape)

inf = bpu_infer_lib.Infer(False)
inf.load_model("/root/rwkv_v5/rwkv_v5_submodel1.bin")

input_index = 0
state_index = 0
state2_index = 0
input_data = np.array([11685], dtype=np.int32)

ret = inf.read_input(state[state_index], input_index)
# print("1:", state[state_index])
# print(type(state[state_index]))
input_index += 1
state_index += 1
ret = inf.read_input(input_data, input_index)
input_index += 1
ret = inf.read_input(state2[state2_index], input_index)
# print("2:", state2[state2_index])
# print(type(state2[state2_index]))
input_index += 1
state2_index += 1

for _ in range(12):
    ret = inf.read_input(state[state_index], input_index)
    input_index += 1
    state_index += 1
    ret = inf.read_input(state[state_index], input_index)
    input_index += 1
    state_index += 1
    ret = inf.read_input(state2[state2_index], input_index) # 如果这里改为state[state2_index]则shape会不同，此时结果不对，但不会报错！！！
    input_index += 1
    state2_index += 1
#print("input_index: ", input_index)
#print("state_index: ", state_index)
#print("state2_index: ", state2_index)

#print("start forward")
inf.forward(True)
#print("end forward")

# res = inf.get_infer_res_np_float32(0, 1024)
# print("res:", res)

inf.get_output()
subgraph1_out1 = inf.outputs[0].data.reshape(1024)
print("subgraph1_out1:", subgraph1_out1[:10])
subgraph1_out2 = inf.outputs[1].data.reshape(1024)
print("subgraph1_out2:", subgraph1_out2[:10])
subgraph1_state = [inf.outputs[j].data.reshape(1024) for j in range(2,27)]
subgraph1_state2 = [inf.outputs[k].data.reshape(16,64,64) for k in range(27,40)]