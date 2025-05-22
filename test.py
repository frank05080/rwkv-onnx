SAVE_INPUT = False
INPUT_FOLDER_NAME = "dumped_inputs"
INPUT_INDEX = 0


def initONNXFile(path, model_version, useAllAvailableProviders=False):
    import onnxruntime as rt

    # session execution provider options
    sess_options = rt.SessionOptions()
    # sess_options.enable_profiling = True

    print(rt.get_available_providers())
    if(not useAllAvailableProviders):
        import inquirer
    providers = inquirer.checkbox(
        "Select execution providers(use space bar to select checkboxes)", choices=rt.get_available_providers()) if not useAllAvailableProviders else rt.get_available_providers()
    print(providers)
    # sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 6
    sess_options.inter_op_num_threads = 6
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

    sess = rt.InferenceSession(
        path, sess_options, providers=providers)

    ins = {
    }

    embed = int(path.split("_")[2].split(".")[0])
    layers = int(path.split("_")[1])
    typenum = sess.get_inputs()[1].type
    print("type of typenum: ", type(typenum))
    print(typenum, embed, layers)
    import numpy as np

    if typenum == "tensor(float)":
        typenum = np.float32
    elif typenum == "tensor(float16)":
        typenum = np.float16

    class InterOp():
        RnnOnly = True

        def forward(selff, xi, statei, statei2):
            # print(statei[0][23])
            # create inputs
            inputs = ins
            # get input names
            input_names = sess.get_inputs()
            input_names = [x.name for x in input_names]
            # get output names
            output_names = sess.get_outputs()
            output_names = [x.name for x in output_names]
            # print("output_names:", output_names)
            # print("len(output_names): ", len(output_names))

            # create input dict
            inputs[input_names[0]] = np.array([xi], dtype=np.int32)
            for i in range(len(input_names)-1):
                # print(input_names[i+1])
                if "wkv" in input_names[i+1]:
                    inputs[input_names[i+1]] = statei2[i-statei.__len__()] # statei2 has shape (24,16,64,64)
                else:
                    # print(i, statei.__len__())
                    inputs[input_names[i+1]] = statei[i] # statei has shape [48, 1024]
                    
            if SAVE_INPUT:
                import os
                if not os.path.exists(INPUT_FOLDER_NAME):
                    os.makedirs(INPUT_FOLDER_NAME)
                global INPUT_INDEX
                for name, array in inputs.items():
                    type_folder = os.path.join(INPUT_FOLDER_NAME, name)
                    os.makedirs(type_folder, exist_ok=True)
                    print(type_folder)
                    file_path = os.path.join(type_folder, f"{str(INPUT_INDEX)}.bin")
                    print(INPUT_INDEX)
                    print(file_path)
                    INPUT_INDEX += 1
                    array.tofile(file_path)

            outputs = sess.run(output_names, inputs)
            # print(outputs[1][23])

            return outputs[0], outputs[1:statei.__len__()+1], outputs[statei.__len__()+1:]
        
    model = InterOp()

    # emptyState = []
    emptyState = np.array(([[0.01]*embed, [0.01]*embed])*layers, typenum)
    head_num = 0
    if model_version == "0.4B":
        head_num = 16
    emptyState2 = np.array(([[[[0.01]*64]*64]*head_num])*layers, typenum) # revise 16
    print (emptyState.shape)
    print (emptyState2.shape)

    return model, emptyState, emptyState2


def npsample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
    import numpy as np
    from scipy.special import softmax

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


def main():
    import inquirer
    # get all .onnx files in current directory
    import os
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    files = [f for f in files if f.endswith(".onnx") or f.endswith(".ort")]
    model, state, state2 = initONNXFile(inquirer.list_input("Select model", choices=files), "0.4B") 

    from tokenizer import world as tokenizer

    # prompt = tokenizer.encode("### Instruction:\nPlease write a short story of a man defeating a two headed dragon###Result\n")
    prompt = tokenizer.encode("### Instruction:\n请解释黑洞###Result\n")
    import tqdm
    for token in tqdm.tqdm(prompt[:-1]):
        logits, state, state2 = model.forward(token, state, state2)

    print("Loaded prompt.")

    for i in range(100):
        logits, state, state2 = model.forward(prompt[-1], state, state2)
        prompt = prompt+[npsample(logits)]
        print(tokenizer.decode(prompt[-1:]),end="", flush=True)
    print(tokenizer.decode(prompt))


if __name__ == "__main__":
    main()