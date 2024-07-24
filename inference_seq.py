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

# # Load the subgraph2 model
# session2 = ort.InferenceSession('subgraph2.onnx')

# # Get input and output names for subgraph2
# input_name2 = session2.get_inputs()[0].name
# input_name3 = session2.get_inputs()[1].name  # This is input2 from the original model
# output_name2 = session2.get_outputs()[0].name

# # Create input data for subgraph2
# input_data2 = np.array([2.0], dtype=np.float32)  # This corresponds to input2 in the original model

# # Run inference on subgraph2 using the output from subgraph1 as one of the inputs
# results2 = session2.run(
#     [output_name2],
#     {input_name2: subgraph1_output, input_name3: input_data2}
# )

# # Get the output from subgraph2
# subgraph2_output = results2[0]
# print(f'Subgraph2 Output: {subgraph2_output}')
