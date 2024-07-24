import onnxruntime as ort
import numpy as np

# Load the subgraph1 model
session1 = ort.InferenceSession('submodel1.onnx')

# Get input and output names for subgraph1
input_name1 = session1.get_inputs()[0].name
output_name1 = session1.get_outputs()[0].name

# Create input data for subgraph1
input_data1 = np.array([200], dtype=np.int32)

# Run inference on subgraph1
results1 = session1.run(
    [output_name1],
    {input_name1: input_data1}
)

# Get the output from subgraph1
subgraph1_output = results1[0]
print(f'Subgraph1 Output: {subgraph1_output}')
print(subgraph1_output.shape)

# ---------- Load the subgraph2 model ------------
session2 = ort.InferenceSession('submodel2.onnx')

input_names2 = session2.get_inputs()
input_names2 = [x.name for x in input_names2]
inputs2 = {}
inputs2[input_names2[0]] = subgraph1_output
embed = 1024
layers = 24
typenum = np.float32
emptyState = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum)
emptyState2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16
for i in range(len(input_names2)-1):
    # print(input_names[i+1])
    if "wkv" in input_names2[i+1]:
        inputs2[input_names2[i+1]] = emptyState2[i-emptyState.__len__()] # statei2 has shape (24,16,64,64)
    else:
        # print(i, statei.__len__())
        inputs2[input_names2[i+1]] = emptyState[i] # statei has shape [48, 1024]

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
