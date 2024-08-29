import numpy as np
# import onnxruntime as ort
from horizon_tc_ui import HB_ONNXRuntime as ort
from scipy.special import softmax

import os
input_folder_name = 'dumped_inputs_submodel1_bak'
if not os.path.exists(input_folder_name):
    os.makedirs(input_folder_name)
index1 = 0
SAVE_INPUT = True

class Model:
    
    def __init__(self, submodel1_path, submodel2_path) -> None:
        # self.session1 = ort.InferenceSession(submodel1_path)
        # self.session2 = ort.InferenceSession(submodel2_path)
        print("start loading submodel1..")
        self.session1 = ort(submodel1_path)
        print("start loading submodel2..")
        self.session2 = ort(submodel2_path)
        print("finish loading all models")
        
    def forward(self, token, state, state2): # state/state2 first is a numpy array, then is a list of np array
        print("token is:", token)
        
        input_names1 = self.session1.get_inputs() # [name.name for name in input_names1]
        input_names1 = [x.name for x in input_names1]
        
        inputs1 = {}
        input_data1 = np.array([token], dtype=np.int32)
        inputs1[input_names1[0]] = input_data1
        
        for i in range(len(input_names1)-1):
            if "wkv" in input_names1[i+1]:
                inputs1[input_names1[i+1]] = state2[i-25] # statei2 has shape (24,16,64,64)
            else:
                inputs1[input_names1[i+1]] = state[i] # statei has shape [48, 1024]
                
        if SAVE_INPUT:
            global index1
            for name, array in inputs1.items():
                type_folder = os.path.join(input_folder_name, name)
                os.makedirs(type_folder, exist_ok=True)
                file_path = os.path.join(type_folder, f"{name + str(index1)}.bin")
                index1 += 1
                array.tofile(file_path)

        output_names1 = self.session1.get_outputs()
        output_names1 = [x.name for x in output_names1]
        # Run inference on subgraph1
        results1 = self.session1.run(
            output_names1,
            inputs1
        )
        # print(output_names1)
        subgraph1_out1 = results1[0]
        print("subgraph1_out1:", subgraph1_out1[:10])
        subgraph1_out2 = results1[1]
        print("subgraph1_out2:", subgraph1_out2[:10])
        subgraph1_state = results1[2:27]
        subgraph1_state2 = results1[-13:]
        
        # # ---------- Load the subgraph2 model ------------
        input_names2 = self.session2.get_inputs()
        input_names2 = [x.name for x in input_names2]
        inputs2 = {}
        inputs2[input_names2[0]] = subgraph1_out1
        inputs2[input_names2[1]] = subgraph1_out2

        for i in range(len(input_names2)-2): # minus matvec_1532_out and add_1480_out
            if "wkv" in input_names2[i+2]:
                inputs2[input_names2[i+2]] = state2[i-10] # statei2 has shape (24,16,64,64)
            else:
                inputs2[input_names2[i+2]] = state[i+25] # statei has shape [48, 1024]

        output_names2 = self.session2.get_outputs()
        output_names2 = [x.name for x in output_names2]
        # print("output_names2:", output_names2)
        results2 = self.session2.run(
            output_names2,
            inputs2
        )
        
        subgraph2_state = results2[1:24]
        subgraph2_state2 = results2[-11:]
        
        return results2[0], subgraph1_state + subgraph2_state, subgraph1_state2 + subgraph2_state2


def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8, seed: int = None) -> int:
    if seed is not None:
        np.random.seed(seed)
    try:
        ozut = ozut.numpy()
    except:
        try:
            ozut = ozut.cpu().numpy()
        except:
            ozut = np.array(ozut)
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # out[self.UNKNOWN_CHAR] = -float('Inf')
    # turn to float if is half and cpu
    probs = softmax(ozut, axis=-1)

    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = pow(probs, 1.0 / temp)
    probs = probs / np.sum(probs, axis=0)
    mout = np.random.choice(a=len(probs), p=probs)
    print("mout is:", mout)
    return mout

from tokenizer import world as tokenizer

embed = 1024
layers = 24
typenum = np.float32
state = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum)
state2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16

# submodel1_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/submodel1.onnx"
submodel1_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/rwkv_v5_submodel1_quantized_model.onnx"
submodel2_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/submodel2.onnx"
# submodel2_path = "/home/ros/share_dir/gitrepos/rwkv-onnx/rwkv_v5_submodel2_quantized_model.onnx"
model = Model(submodel1_path, submodel2_path)

# prompt = tokenizer.encode("### Instruction:\n晚上吃什么###Result\n")
# prompt = tokenizer.encode("###Question\n 我想听一个故事 ###Answer\n")
prompt = tokenizer.encode("你好")
import tqdm
for token in tqdm.tqdm(prompt[:-1]):
    logits, state, state2 = model.forward(token, state, state2)
    print("logits[:10]:", logits[:10])
    print("state[0][:10]: ", state[0][:10])
    print("state2[0][:10]: ", state2[0][0][0][:10])
print("Loaded prompt.")

for i in range(1000):
    logits, state, state2 = model.forward(prompt[-1], state, state2)
    print("logits[:10]:", logits[:10])
    print("state[0][:10]: ", state[0][:10])
    print("state2[0][:10]: ", state2[0][0][0][:10])
    prompt = prompt+[npsample(logits, seed=42)]
    print(tokenizer.decode(prompt[-1:]),end="", flush=True)
print(tokenizer.decode(prompt))