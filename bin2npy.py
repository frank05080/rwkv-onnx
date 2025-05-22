import os
import numpy as np

# 配置参数
# input_root = "C:\\Users\\guanzhong.chen\\Documents\\virtualbox_share\\gitrepos\\rwkv-onnx\\J6toolchain\\dumped_input_one\\dumped_inputs"
input_root = "C:\\Users\\guanzhong.chen\\Documents\\virtualbox_share\\gitrepos\\rwkv-onnx\\dumped_inputs"
output_root = "npy_folder"       # 输出根目录
os.makedirs(output_root, exist_ok=True)

def convert_and_save(bin_path, npy_path, shape, dtype):
    data = np.fromfile(bin_path, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"数据大小不匹配: {bin_path} -> {data.size} 元素, 期望 {np.prod(shape)}")
    data = data.reshape(shape)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, data)
    print(f"✅ 已保存: {npy_path}")

def process_all(input_root, output_root):
    for dirpath, dirnames, filenames in os.walk(input_root):
        # print(filenames)
        # print(dirpath)
        bin_files = [f for f in filenames if f.lower().endswith('.bin')]
        print(bin_files)
        subfold_name = dirpath.split('\\')[-1] # input0
        new_folder = os.path.join(output_root, subfold_name)
        os.makedirs(new_folder, exist_ok=True)
        if len(bin_files) > 0:
            for bin_file in bin_files:
                old_path = os.path.join(dirpath, bin_file)
                npy_name = bin_file.split(".")[0] + ".npy"
                new_path = os.path.join(new_folder, npy_name)
                print(new_path)
                
                shape = None
                dtype = None
                if 'input0' in new_path:
                    shape = (1)
                    dtype = np.int32
                elif 'instatewkv' in new_path:
                    shape = (16, 64, 64)
                    dtype = np.float32
                else:
                    shape = (1024)
                    dtype = np.float32

                try:
                    convert_and_save(old_path, new_path, shape, dtype)
                except Exception as e:
                    print(f"❌ 错误处理 {old_path} -> {e}")

if __name__ == "__main__":
    process_all(input_root, output_root)
