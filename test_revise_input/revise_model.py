import onnx
import onnx.helper as helper

model_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/initial_model.onnx"
model = onnx.load(model_path)

# # Update the opset version to 11
# model.opset_import[0].version = 11

graph = model.graph
nodes = list(graph.node)

## Find the input and then remove it
old_input_name = 'input1'
for i, input in enumerate(graph.input):
    if input.name == old_input_name:
        del graph.input[i]
        break

new_input_name = 'input0'  # should not be equal to the old input name
new_input = helper.make_tensor_value_info(new_input_name, onnx.TensorProto.FLOAT, [1, 1])

# Instead of a tensor, create a constant node for the reshape shape
reshape_shape = helper.make_tensor(
    name='reshape_shape_const',
    data_type=onnx.TensorProto.INT64,
    dims=[1],
    vals=[1]
)
reshape_shape_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['reshape_shape'],
    value=reshape_shape
)

reshape_output_name = '/Reshaped_Output'
reshape_node = helper.make_node(
    'Reshape',
    inputs=[new_input_name, 'reshape_shape'],
    outputs=[reshape_output_name],
    name='Reshape_New_Input'
)

# Ensure the reshape nodes are added before any other node that uses its output
new_nodes = [reshape_shape_node, reshape_node] # reshape node appears before other nodes

# Update the rest of the graph to use the new reshaped output instead of the old input
for node in nodes:
    for i, input_name in enumerate(node.input):
        if input_name == old_input_name:
            node.input[i] = reshape_output_name
    new_nodes.append(node)  # Add nodes in correct order

# Replace the existing nodes with the new sorted nodes ## First clear graph.node field then fill in 
graph.ClearField('node') # can not use graph.node.clear() since 'google.protobuf.pyext._message.RepeatedCompositeCo' object has no attribute 'clear'
graph.node.extend(new_nodes)

# Add the new input to the graph
# graph.input.extend([new_input]) 
graph.input.insert(0, new_input) # insert into the first position to maintain the input order

new_model = helper.make_model(graph)

# change opset 17 to opset 11
for opset in new_model.opset_import:
    print(f"Domain: {opset.domain}, Version: {opset.version}")
    if opset.domain == '' or opset.domain == 'ai.onnx':
        opset.version = 11  # Change to 10 if needed
        print(f"Domain: {opset.domain}, Version: {opset.version}")

# Save the modified model
onnx.save(new_model, "/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/revised_graph.onnx")

# Load the modified model to perform the check
try:
    onnx_model = onnx.load("/home/ros/share_dir/gitrepos/rwkv-onnx/test_revise_input/revised_graph.onnx")
    onnx.checker.check_model(onnx_model)
    
    for opset in new_model.opset_import:
        print(f"Domain: {opset.domain}, Version: {opset.version}")
    
    # onnx.save(new_model, "/home/ros/share_dir/gitrepos/rwkv_v4/data/v4/pt2onnx_models/revised_input_body.onnx")
    # print("Model checked successfully")
except Exception as e:
    print(f"Model check failed: {e}")
