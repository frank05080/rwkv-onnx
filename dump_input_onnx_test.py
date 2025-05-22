## Docker内跑

import os
import numpy as np
from horizon_tc_ui.hb_runtime import HBRuntime as ort
import shutil

SOLELY_SAVE_DUMP = True

# 1. 定义模型路径和输入数据路径
model_path = "/j6_oe/rwkv_v5/model_output/RWKV_24_1024_32_11_optimized_float_model.onnx"  # 请根据你的实际路径修改
input_root = "/j6_oe/rwkv_v5/dumped_inputs"
new_create_dump_input_folder = "selected_output"
# Create output directory if it doesn't exist
os.makedirs(new_create_dump_input_folder, exist_ok=True)

# 2. 加载 ONNX 模型
session = ort(model_path)
input_names = [inp for inp in session.input_names]
print(f"模型输入名：{input_names}")
output_names = session.output_names

# 3. 从每个子文件夹中读取一个 .bin 文件
onnx_inputs = {}
for name in input_names:
    folder = os.path.join(input_root, name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"未找到输入文件夹：{folder}")

    # 获取第一个 .bin 文件
    files = [f for f in os.listdir(folder) if f.endswith('.bin')]
    if not files:
        raise FileNotFoundError(f"{folder} 中未找到 .bin 文件")

    bin_path = os.path.join(folder, files[0])
    print("bin_path:", bin_path)
    
    if("input0" in bin_path):
        array = np.fromfile(bin_path, dtype=np.int32).reshape((1))
        onnx_inputs[name] = array
    elif("instatewkv" in bin_path):
        array = np.fromfile(bin_path, dtype=np.float32).reshape((16, 64, 64))
        onnx_inputs[name] = array
    else:
        array = np.fromfile(bin_path, dtype=np.float32).reshape((1024))
        onnx_inputs[name] = array
        
    # copy to new location
    if SOLELY_SAVE_DUMP:
        new_folder = os.path.join(new_create_dump_input_folder, name)
        os.makedirs(new_folder, exist_ok=True)
        shutil.copy(bin_path, os.path.join(new_folder, files[0]))
        print(f"Copied {files[0]} to {new_folder}")

# 4. 推理
outputs = session.run(output_names, onnx_inputs)

# 5. 显示输出
for i, output in enumerate(outputs):
    print(f"输出{i}: shape={output.shape}")
