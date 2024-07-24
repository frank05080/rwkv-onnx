import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession('initial_model.onnx')

# Get input and output names
input_name1 = session.get_inputs()[0].name
input_name2 = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name

# Create input data
input_data1 = np.array([1.0], dtype=np.float32)
input_data2 = np.array([2.0], dtype=np.float32)

# Run inference
results = session.run(
    [output_name], 
    {input_name1: input_data1, input_name2: input_data2}
)

# Print the result
print(f'Output: {results[0]}')
