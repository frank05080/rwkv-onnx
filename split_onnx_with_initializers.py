from onnx import helper, shape_inference, checker, TensorProto
import copy
import onnx

## Initializers are constant tensors used in the graph.

def find_node_index(nodes, node_name):
    for i, node in enumerate(nodes):
        if node.name == node_name:
            split_index = i
            print("split_index: ", split_index)
            return split_index
    return -1

# Get used inputs and initializers for subgraph1
def get_used_inputs_and_initializers(nodes, graph):
    used_inputs = set()
    used_initializers = []
    for node in nodes:
        for input_name in node.input:
            used_inputs.add(input_name)
            # Check if the input is an initializer
            for initializer in graph.initializer:
                if initializer.name == input_name:
                    print("initializer.name: ", initializer.name)
                    used_initializers.append(initializer)
    return used_inputs, used_initializers

def get_used_outputs_and_initializers(nodes):
    used_output = set()
    for node in nodes:
        for output_name in node.output:
            used_output.add(output_name)
    return used_output


def split_onnx_model(model, split_node_name, first_subgraph_second_output_node_name):
    split_found = False
    graph = model.graph
    # for output in graph.output:
    #     print(output.name)
    nodes = list(graph.node)
    subgraph1_nodes = []
    subgraph2_nodes = []
    for node in nodes:
        if node.name == split_node_name:
            split_found = True
        if not split_found:
            subgraph1_nodes.append(node)
        else:
            subgraph2_nodes.append(node)
    # print(subgraph1_nodes)

    subgraph1_output_name = subgraph1_nodes[-1].output[0]
    print("subgraph1_output_name: ", subgraph1_output_name)
    split_index = find_node_index(nodes, first_subgraph_second_output_node_name)
    split_node = nodes[split_index]
    subgraph1_output_name2 = split_node.output[0]
    print("subgraph1_output_name2: ", subgraph1_output_name2)

    new_output = helper.make_tensor_value_info(
        subgraph1_output_name, TensorProto.FLOAT, [1024]
    )
    new_output2 = helper.make_tensor_value_info(
        subgraph1_output_name2, TensorProto.FLOAT, [1024]
    )
    
    used_output_subgraph1 = (
        get_used_outputs_and_initializers(subgraph1_nodes)
    )
    subgraph1_outputs = [
        graph_output
        for graph_output in graph.output
        if graph_output.name in used_output_subgraph1
    ] # need to add used outputs!!!
    # print("used_output_subgraph1: ", used_output_subgraph1)
    # print("subgraph1_outputs: ", subgraph1_outputs)
    
    used_inputs_subgraph1, used_initializers_subgraph1 = (
        get_used_inputs_and_initializers(subgraph1_nodes, graph)
    )
    subgraph1_inputs = [
        graph_input
        for graph_input in graph.input
        if graph_input.name in used_inputs_subgraph1
    ]
    
    subgraph1_outputs = [new_output, new_output2] + subgraph1_outputs
    # print("subgraph1_outputs: ", subgraph1_outputs)
    subgraph1 = helper.make_graph(
        subgraph1_nodes, 
        "subgraph1", 
        subgraph1_inputs,
        subgraph1_outputs,
        initializer=used_initializers_subgraph1
    )
    
    # ------------ Start Dealing with Graph2 --------------
    # print(subgraph2_nodes[:3])
    # print(len(graph.input))
    new_input = helper.make_tensor_value_info(
        subgraph1_output_name, TensorProto.FLOAT, [1024]
    )
    new_input2 = helper.make_tensor_value_info(
        subgraph1_output_name2, TensorProto.FLOAT, [1024]
    )
    print(subgraph2_nodes[0])
    subgraph2_nodes[0].input[0] = subgraph1_output_name
    subgraph2_nodes[0].input[1] = subgraph1_output_name2
    used_inputs_subgraph2, used_initializers_subgraph2 = get_used_inputs_and_initializers(subgraph2_nodes, graph)
    subgraph2_inputs = [
        graph_input
        for graph_input in graph.input
        if graph_input.name in used_inputs_subgraph2
    ]
    subgraph2_inputs = [new_input, new_input2] + subgraph2_inputs
    # print("subgraph2_inputs: ", [inpt.name for inpt in subgraph2_inputs])
    
    used_output_subgraph2 = (
        get_used_outputs_and_initializers(subgraph2_nodes)
    )
    subgraph2_outputs = [
        graph_output
        for graph_output in graph.output
        if graph_output.name in used_output_subgraph2
    ]
    subgraph2 = helper.make_graph(
        subgraph2_nodes,
        "subgraph2",
        subgraph2_inputs,  # Including the original input2
        subgraph2_outputs,
        initializer=used_initializers_subgraph2
    )
    
    model_subgraph1 = helper.make_model(subgraph1, opset_imports=[helper.make_opsetid("", 11)])
    model_subgraph2 = helper.make_model(subgraph2, opset_imports=[helper.make_opsetid("", 11)])

    return model_subgraph1, model_subgraph2


# Split the model at the specified node
split_node_name = "Add_988" # revise
model = onnx.load("/home/ros/share_dir/gitrepos/rwkv-onnx/modified_model.onnx")
submodel1, submodel2 = split_onnx_model(model, split_node_name, "Add_935") # revise - find in the original model
# submodel1 = split_onnx_model(model, split_node_name, "Add_11")

# Save the sub-models
onnx.save(submodel1, "submodel1.onnx")
onnx.save(submodel2, "submodel2.onnx")
