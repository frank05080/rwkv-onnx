### Model Checked Fail

import onnx
import onnx.helper as helper

model_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/initial_model.onnx"
model = onnx.load(model_path)

graph = model.graph
# for output in graph.output:
#     print(output.name)
nodes = list(graph.node)
# print(nodes)

old_input_name = 'input1'
# Find the old input and remove it
for i, input in enumerate(graph.input):
    if input.name == old_input_name:
        del graph.input[i]
        break
print(graph.input)

new_input_name = 'input0' # should not be equal to the old input name
new_input = helper.make_tensor_value_info(new_input_name, onnx.TensorProto.FLOAT, [1, 1])

# # Add a squeeze layer to transform [1, 1] into [1]
# squeeze_output_name = 'squeezed_output'
# squeeze_node = helper.make_node(
#     'Squeeze',
#     inputs=[new_input_name],
#     outputs=[squeeze_output_name],
#     name='Squeeze_New_Input',
#     # axes=[0]
# )
# graph.node.append(squeeze_node)
# # Update the rest of the graph to use the new squeezed output instead of the old input
# for node in graph.node:
#     for i, input_name in enumerate(node.input): # 遍历所有节点的input name
#         if input_name == old_input_name: 
#             node.input[i] = squeeze_output_name


# Add a reshape layer to transform [1, 1] into [1]
reshape_shape_name = 'reshape_shape'
reshape_shape = helper.make_tensor(
    reshape_shape_name, onnx.TensorProto.INT64, [1], [1]
)
reshape_output_name = 'reshaped_output'
reshape_node = helper.make_node(
    'Reshape',
    inputs=[new_input_name, reshape_shape_name],
    outputs=[reshape_output_name],
    name='Reshape_New_Input'
)
graph.node.append(reshape_node)

# Update the rest of the graph to use the new squeezed output instead of the old input
for node in graph.node:
    for i, input_name in enumerate(node.input):  # Traverse all nodes' input names
        if input_name == old_input_name: 
            node.input[i] = reshape_output_name


# Add the new input to the graph
graph.input.extend([new_input])
graph.initializer.extend([reshape_shape])

new_model = helper.make_model(graph)

new_graph = new_model.graph
# for output in graph.output:
#     print(output.name)
nodes = list(new_graph.node)

onnx.checker.check_model(new_model)
"""
model checked fail reason: `exp` node uses `reshaped_output` as input,
that is the output of `Reshape_New_Input` node,
but topologically, `Reshape_New_Input` appears after `exp` node..
[
    input: "reshaped_output"
    output: "exp_output"
    name: "exp"
    op_type: "Exp"
, 
    input: "exp_output"
    output: "exp_1_output"
    name: "exp_1"
    op_type: "Exp"
, 
    input: "exp_1_output"
    output: "exp_2_output"
    name: "exp_2"
    op_type: "Exp"
, 
    input: "exp_2_output"
    input: "input2"
    output: "output"
    name: "add"
    op_type: "Add"
, 
    input: "input0"
    input: "reshape_shape"
    output: "reshaped_output"
    name: "Reshape_New_Input"
    op_type: "Reshape"
]
"""
onnx.save(new_model, "/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/revised_graph.onnx")

### Modify the input shape
# for input in graph.input:
#     if input.name == 'input1':
#         """
#         tensor_type {
#             elem_type: 1
#             shape {
#                 dim {
#                     dim_value: 1
#                 }
#             }
#         }
#         """
#         input.type.tensor_type.shape.dim[0].dim_value = 1