import onnxruntime as ort
import numpy as np

## 10 and 25 are manually calculated, fill in all emptystates and emptystates2 sequentilly

# Load the subgraph1 model
session1 = ort.InferenceSession('submodel1.onnx')

input_names1 = session1.get_inputs()
input_names1 = [x.name for x in input_names1]
print(input_names1)
inputs1 = {}
input_data1 = np.array([200], dtype=np.int32)
inputs1[input_names1[0]] = input_data1
embed = 1024
layers = 24
typenum = np.float32
emptyState = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum)
print(len(emptyState), emptyState.__len__())
emptyState2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16
print("len(input_names1): ", len(input_names1)) # 39
for i in range(len(input_names1)-1):
    print("i is:", i)
    # print(input_names[i+1])
    if "wkv" in input_names1[i+1]:
        print("i-len(emptyState):", i-25)
        inputs1[input_names1[i+1]] = emptyState2[i-25] # statei2 has shape (24,16,64,64)
    else:
        # print(i, statei.__len__())
        inputs1[input_names1[i+1]] = emptyState[i] # statei has shape [48, 1024]

# Get input and output names for subgraph1
input_name1 = session1.get_inputs()[0].name
output_name1 = session1.get_outputs()[0].name

output_names1 = session1.get_outputs()
output_names1 = [x.name for x in output_names1]
# Run inference on subgraph1
results1 = session1.run(
    output_names1,
    inputs1
)

subgraph1_output = results1[0]
print(f'Subgraph1 Output: {subgraph1_output}')
print(subgraph1_output.shape)
print(output_names1)

subgraph1_output2 = results1[1]
print(f'Subgraph1 Output2: {subgraph1_output2}')
print(subgraph1_output2.shape)

subgraph1_output3 = results1[2]
print(f'Subgraph1 Output2: {subgraph1_output3}')
print(subgraph1_output3.shape)

# # ---------- Load the subgraph2 model ------------
session2 = ort.InferenceSession('submodel2.onnx')

input_names2 = session2.get_inputs()
input_names2 = [x.name for x in input_names2]
print("input_names2: ", input_names2)
inputs2 = {}
inputs2[input_names2[0]] = subgraph1_output
inputs2[input_names2[1]] = subgraph1_output2

print("len(input_names2):", len(input_names2)) # 36
print("emptyState.shpae:", emptyState.shape)
print("emptyState2.shpae:", emptyState2.shape)
for i in range(len(input_names2)-2): # minus matvec_1532_out and add_1480_out
    print("i is:", i)
    # print(input_names[i+1])
    if "wkv" in input_names2[i+2]:
        print("i-10 is:", i-10)
        inputs2[input_names2[i+2]] = emptyState2[i-10] # statei2 has shape (24,16,64,64)
    else:
        # print(i, statei.__len__())
        print(i+25)
        inputs2[input_names2[i+2]] = emptyState[i+25] # statei has shape [48, 1024]

output_names2 = session2.get_outputs()
output_names2 = [x.name for x in output_names2]
# Run inference on subgraph1
results2 = session2.run(
    output_names2,
    inputs2
)

subgraph2_output = results2[0]
print(f'Subgraph2 Output: {subgraph2_output}')
print(subgraph2_output.shape)
