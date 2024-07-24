import onnxruntime as ort
import numpy as np

# Load the subgraph1 model
session1 = ort.InferenceSession('/home/ros/share_dir/gitrepos/rwkv-onnx/RWKV_24_1024_32_11_full.onnx')

# Get input and output names for subgraph1
input_name1 = session1.get_inputs()[0].name
output_name1 = session1.get_outputs()[0].name

# Create input data for subgraph1
embed = 1024
layers = 24
typenum = np.float32
input_data1 = np.array([200], dtype=np.int32)
emptyState = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum)
emptyState2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16

inputs = {}
input_names = session1.get_inputs()
input_names = [x.name for x in input_names]
inputs[input_names[0]] = input_data1
for i in range(len(input_names)-1):
    # print(input_names[i+1])
    if "wkv" in input_names[i+1]:
        inputs[input_names[i+1]] = emptyState2[i-emptyState.__len__()] # statei2 has shape (24,16,64,64)
    else:
        # print(i, statei.__len__())
        inputs[input_names[i+1]] = emptyState[i] # statei has shape [48, 1024]

output_names = session1.get_outputs()
output_names = [x.name for x in output_names]
# Run inference on subgraph1
results1 = session1.run(
    output_names,
    inputs
)

# Get the output from subgraph1
subgraph1_output = results1[0]
print(f'Subgraph1 Output: {subgraph1_output}')
print(subgraph1_output.shape)