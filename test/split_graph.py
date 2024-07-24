import onnx
from onnx import helper, TensorProto

def get_used_inputs(nodes):
    used_inputs = set()
    for node in nodes:
        for input_name in node.input:
            used_inputs.add(input_name)
    return used_inputs

# Load the existing ONNX model
model = onnx.load("/home/ros/share_dir/gitrepos/rwkv-onnx/test/initial_model.onnx")
graph = model.graph

# Find the nodes to split the graph
nodes = list(graph.node)

# Identify the nodes in the two subgraphs
subgraph1_nodes = []
subgraph2_nodes = []
split_node = "exp_2"  # The node where you want to split

# Iterate through nodes to classify them into subgraph1 and subgraph2
split_found = False
for node in nodes:
    if node.name == split_node:
        split_found = True
    if not split_found:
        subgraph1_nodes.append(node)
    else:
        subgraph2_nodes.append(node)

# Create new output node for subgraph1
subgraph1_output_name = subgraph1_nodes[-1].output[0]
new_output = helper.make_tensor_value_info(
    subgraph1_output_name, TensorProto.FLOAT, [1]
)

for node in subgraph1_nodes:
    for input in node.input:
        print(input)

for graph_input in graph.input:
    print(graph_input)

# Create the subgraph1
# used_inputs_subgraph1 = get_used_inputs(subgraph1_nodes)
subgraph1_inputs = [
    graph_input
    for graph_input in graph.input
    if graph_input.name
    in [input for node in subgraph1_nodes for input in node.input]
] # filter graph inputs, if they are in subgraph inputs!!!
subgraph1 = helper.make_graph(subgraph1_nodes, "subgraph1", subgraph1_inputs, [new_output]) # graph.input

# Create new input node for subgraph2
new_input = helper.make_tensor_value_info(subgraph1_output_name, TensorProto.FLOAT, [1])

# Update the inputs of the first node in subgraph2
subgraph2_nodes[0].input[0] = subgraph1_output_name

# Create the subgraph2
subgraph2 = helper.make_graph(
    subgraph2_nodes,
    "subgraph2",
    [new_input, graph.input[1]],  # Including the original input2 # never forget to include graph.input[1] here!!! - input2
    graph.output,
)

# Create new models for each subgraph
model_subgraph1 = helper.make_model(subgraph1)
model_subgraph2 = helper.make_model(subgraph2)

# Save the new models
onnx.save(model_subgraph1, "subgraph1.onnx")
onnx.save(model_subgraph2, "subgraph2.onnx")

print("Subgraph1 and Subgraph2 models created and saved successfully.")
