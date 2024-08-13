import numpy as np
from scipy.special import softmax
import onnxruntime as ort

import bpu_infer_lib


class Model:
    def __init__(self, submodel1_path, submodel2_path) -> None:
        self.inf = bpu_infer_lib.Infer(False)
        print("1111111111111ready to load model")
        self.inf.load_model(submodel1_path)
        self.session2 = ort.InferenceSession(submodel2_path)
        
    def forward(self, token, state, state2):
        
        # self.inf = bpu_infer_lib.Infer(False)
        # self.inf.load_model(submodel1_path)
        
        input_index = 0
        state_index = 0
        state2_index = 0
        input_data = np.array([token], dtype=np.int32)
        
        ret = self.inf.read_numpy_arr_float32(state[state_index], input_index)
        input_index += 1
        state_index += 1
        ret = self.inf.read_numpy_arr_int32(input_data, input_index)
        input_index += 1
        ret = self.inf.read_numpy_arr_float32(state2[state2_index], input_index)
        input_index += 1
        state2_index += 1
        
        for _ in range(12):
            ret = self.inf.read_numpy_arr_float32(state[state_index], input_index)
            input_index += 1
            state_index += 1
            ret = self.inf.read_numpy_arr_float32(state[state_index], input_index)
            input_index += 1
            state_index += 1
            ret = self.inf.read_numpy_arr_float32(state2[state2_index], input_index) # 如果这里改为state[state2_index]则shape会不同，此时结果不对，但不会报错！！！
            input_index += 1
            state2_index += 1
        #print("input_index: ", input_index)
        #print("state_index: ", state_index)
        #print("state2_index: ", state2_index)
        
        #print("start forward")
        self.inf.forward(True)
        #print("end forward")
        
        # res = self.inf.get_infer_res_np_float32(0, 1024)
        # print("res:", res)
        
        subgraph1_out1 = self.inf.get_infer_res_np_float32(0, 1024)
        subgraph1_out2 = self.inf.get_infer_res_np_float32(1, 1024)
        subgraph1_state = [self.inf.get_infer_res_np_float32(j, 1024) for j in range(2,27)]
        subgraph1_state2 = [self.inf.get_infer_res_np_float32(k, 65536).reshape(16,64,64) for k in range(27,40)]
        
        # del (self.inf)
                
        # ---------- Load the subgraph2 model ------------
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


def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
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
    return mout

from tokenizer import world as tokenizer

embed = 1024
layers = 24
typenum = np.float32
state = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum) # [48, 1024]
state2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16 # [24, 16, 64, 64]

submodel1_path = "/root/rwkv_v5/rwkv_v5_submodel1.bin"
submodel2_path = "/root/rwkv_v5/submodel2.onnx"
model = Model(submodel1_path, submodel2_path)

# prompt = tokenizer.encode("### Instruction:\n请问你是谁###Result\n")
# prompt = tokenizer.encode("请介绍黑洞：")
prompt = tokenizer.encode("###Question\n 解释黑洞 ###Answer\n")

import tqdm
for token in tqdm.tqdm(prompt[:-1]):
    logits, state, state2 = model.forward(token, state, state2)
print("Loaded prompt.")

for i in range(1000):
    logits, state, state2 = model.forward(prompt[-1], state, state2)
    prompt = prompt+[npsample(logits)]
    print(tokenizer.decode(prompt[-1:]),end="", flush=True)
print(tokenizer.decode(prompt))
