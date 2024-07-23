import onnx

# Load the ONNX model
model_path = '/home/ros/share_dir/gitrepos/rwkv-onnx/opset11_onnx/RWKV_24_1024_32_11_full.onnx'
model = onnx.load(model_path)

# Access the graph's nodes
nodes = model.graph.node

# Set or modify node names based on their type
for i, node in enumerate(nodes):
    # Set the node name to its type followed by its index
    node.name = f"{node.op_type}_{i}"

# Save the modified model if needed
onnx.save(model, 'modified_model.onnx')