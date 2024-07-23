from onnx import helper, shape_inference, checker, TensorProto
import copy
import onnx

def split_onnx_model(model, split_node_name):
    # Find the split node
    nodes = model.graph.node
    split_index = None
    for i, node in enumerate(nodes):
        print(node.name)
        if node.name == split_node_name:
            split_index = i
            break

    if split_index is None:
        raise ValueError(f"Node {split_node_name} not found in the model")

    # Create the first sub-model
    subgraph1_nodes = nodes[:split_index + 1]
    subgraph1 = helper.make_graph(
        subgraph1_nodes,
        'subgraph1',
        model.graph.input,
        model.graph.output
    )
    submodel1 = helper.make_model(subgraph1)
    submodel1 = shape_inference.infer_shapes(submodel1)

    # Create the second sub-model
    subgraph2_nodes = nodes[split_index + 1:]
    subgraph2 = helper.make_graph(
        subgraph2_nodes,
        model.graph.input,
        model.graph.output
    )
    submodel2 = helper.make_model(subgraph2)
    submodel2 = shape_inference.infer_shapes(submodel2)

    # Check the sub-models for validity
    checker.check_model(submodel1)
    checker.check_model(submodel2)

    return submodel1, submodel2

# Split the model at the specified node
split_node_name = "name_of_the_split_node"
model = onnx.load("/home/ros/share_dir/gitrepos/rwkv-onnx/opset11_onnx/RWKV_24_1024_32_11.onnx")
submodel1, submodel2 = split_onnx_model(model, split_node_name)

# Save the sub-models
onnx.save(submodel1, "submodel1.onnx")
onnx.save(submodel2, "submodel2.onnx")