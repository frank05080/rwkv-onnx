import onnx
from onnx import helper, TensorProto

# Create the initial graph nodes and tensors
input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [1])
input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [1])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])

# Define the exponential nodes with names
exp_node = helper.make_node(
    'Exp',
    inputs=['input1'],
    outputs=['exp_output'],
    name='exp'
)

exp_1_node = helper.make_node(
    'Exp',
    inputs=['exp_output'],
    outputs=['exp_1_output'],
    name='exp_1'
)

exp_2_node = helper.make_node(
    'Exp',
    inputs=['exp_1_output'],
    outputs=['exp_2_output'],
    name='exp_2'
)

# Define the add node to combine exp_2_output and input2
add_node = helper.make_node(
    'Add',
    inputs=['exp_2_output', 'input2'],
    outputs=['output'],
    name='add'
)

# Create the graph
graph = helper.make_graph(
    [exp_node, exp_1_node, exp_2_node, add_node],
    'example_graph',
    [input1, input2],
    [output]
)

# Create the model
model = helper.make_model(graph, producer_name='example_model')

# Save the model
onnx.save(model, 'initial_model.onnx')

print('Model created and saved as initial_model.onnx')
