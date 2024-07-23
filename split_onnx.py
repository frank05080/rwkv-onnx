from onnx import helper, shape_inference, checker, TensorProto
import copy
import onnx

def get_used_inputs_outputs(nodes):
    used_inputs = set()
    used_outputs = set()
    for node in nodes:
        used_inputs.update(node.input)
        used_outputs.update(node.output)
    return used_inputs, used_outputs

def remove_redundant_io(graph, used_inputs, used_outputs):
    new_inputs = [inp for inp in graph.input if inp.name in used_inputs]
    new_outputs = [outp for outp in graph.output if outp.name in used_outputs]
    return new_inputs, new_outputs

def split_onnx_model(model, split_node_name):
    # Find the split node
    nodes = model.graph.node
    split_index = None
    for i, node in enumerate(nodes):
        if node.name == split_node_name:
            split_index = i
            print("split_index: ", split_index)
            break

    if split_index is None:
        raise ValueError(f"Node {split_node_name} not found in the model")

    # Get the output names of the split node
    split_node = nodes[split_index]
    split_node_outputs = split_node.output

    # Create the first sub-model
    subgraph1_nodes = nodes[: split_index + 1]
    used_inputs_subgraph1, used_outputs_subgraph1 = get_used_inputs_outputs(subgraph1_nodes)
    used_outputs_subgraph1.update(split_node_outputs)
    subgraph1_inputs, subgraph1_outputs = remove_redundant_io(model.graph, used_inputs_subgraph1, used_outputs_subgraph1)
    # for name in split_node_outputs:
    #     print("name: ", name)
    subgraph1_outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        for name in split_node_outputs
    ]
    subgraph1 = helper.make_graph(
        subgraph1_nodes, "subgraph1", subgraph1_inputs, subgraph1_outputs
    ) # model.graph.input
    submodel1 = helper.make_model(subgraph1)
    submodel1 = shape_inference.infer_shapes(submodel1)

    # Create the second sub-model
    subgraph2_nodes = nodes[split_index + 1 :]
    used_inputs_subgraph2, used_outputs_subgraph2 = get_used_inputs_outputs(subgraph2_nodes)
    used_inputs_subgraph2.update(split_node_outputs)
    subgraph2_inputs, subgraph2_outputs = remove_redundant_io(model.graph, used_inputs_subgraph2, used_outputs_subgraph2)
    subgraph2_inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        for name in split_node_outputs
    ]
    subgraph2 = helper.make_graph(
        subgraph2_nodes, "subgraph2", subgraph2_inputs, model.graph.output
    ) # model.graph.output
    submodel2 = helper.make_model(subgraph2)
    submodel2 = shape_inference.infer_shapes(submodel2)

    # Check the sub-models for validity
    # checker.check_model(submodel1)
    # checker.check_model(submodel2)

    return submodel1, submodel2

# Split the model at the specified node
split_node_name = "Mul_10"
model = onnx.load("/home/ros/share_dir/gitrepos/rwkv-onnx/modified_model.onnx")
submodel1, submodel2 = split_onnx_model(model, split_node_name)

# Save the sub-models
onnx.save(submodel1, "submodel1.onnx")
onnx.save(submodel2, "submodel2.onnx")
