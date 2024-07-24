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