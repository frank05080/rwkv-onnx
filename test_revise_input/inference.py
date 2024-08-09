import onnxruntime as ort
import numpy as np

USE_REVISE_MODEL = True

# Load the model
session = None
if USE_REVISE_MODEL:
    session = ort.InferenceSession('/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/revised_graph.onnx')
else:
    session = ort.InferenceSession('/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/initial_model.onnx')

# Get input and output names
input_name1 = session.get_inputs()[0].name # input0
print(input_name1)
input_name2 = session.get_inputs()[1].name # input2
output_name = session.get_outputs()[0].name

# Create input data
input_data1 = None
if USE_REVISE_MODEL:
    input_data1 = np.array([1.0], dtype=np.float32).reshape(1,1)
else:
    input_data1 = np.array([1.0], dtype=np.float32)
input_data2 = np.array([2.0], dtype=np.float32)

# Run inference
results = session.run(
    [output_name], 
    {input_name1: input_data1, input_name2: input_data2}
)

# Print the result
print(f'Output: {results[0]}')
