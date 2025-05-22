import numpy as np
np.random.seed(11)
from scipy.special import softmax
import libmodel_task


class Model:
    def __init__(self, model_path) -> None:
        self.inf = libmodel_task.ModelTask()
        self.inf.ModelInit(model_path)
        
    def forward(self, token, state, state2):
        input_data = []
        input_data.append(np.array([token], dtype=np.int32))
        
        for i in range(48):
            input_data.append(state[i])
            
        for i in range(24):
            input_data.append(state2[i])
        
        print("pre infer")
        print(input_data)
        out = self.inf.ModelInfer(input_data)
        print(out)
        import pdb;pdb.set_trace()
        return out


def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.7) -> int:
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
    #mout = np.argmax(probs)
    #sorted_indices = np.argsort(probs)
    #mout = sorted_indices[-2]
    #print("mout is:", mout)
    return mout


from tokenizer import world as tokenizer

embed = 1024
layers = 24
typenum = np.float32
state = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum) # [48, 1024]
state2 = np.array(([[[[0.01]*64]*64]*16])*layers, typenum) # revise 16 # [24, 16, 64, 64]

model_path = "RWKV_24_1024_32_11.hbm"
model = Model(model_path=model_path)

#prompt = tokenizer.encode("### Question:请介绍黑洞。\n### Answer: ")
prompt = tokenizer.encode("### Question:谈谈数学。\n### Answer: ")
#prompt = tokenizer.encode("请介绍黑洞：")
# prompt = tokenizer.encode("我想听一个故事。")
#prompt = tokenizer.encode("你好")

import tqdm
for token in tqdm.tqdm(prompt[:-1]):
    logits, state, state2 = model.forward(token, state, state2)
print("Loaded prompt.")

for i in range(1000):
    logits, state, state2 = model.forward(prompt[-1], state, state2)
    # print("logits[:10]:", logits[:10])
    # print("type of state:", type(state))
    # print("len of state:", len(state))
    # print("state[0][:10]: ", state[0][:10])
    # print("type of state2:", type(state2))
    # print("len of state2:", len(state2))
    # print("state2[0][:10]: ", state2[0][0][0][:10])
    prompt = prompt+[npsample(logits)]
    print(tokenizer.decode(prompt[-1:]),end="", flush=True)
print(tokenizer.decode(prompt))
