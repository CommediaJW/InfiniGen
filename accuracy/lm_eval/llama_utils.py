import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import os
from fastchat.model import get_conversation_template

def set_symlink(model_type, fname):
    model_path = "/nfs/jwyi/workspace/kvclus/infinigen_venv/lib/python3.8/site-packages/transformers/models/" + model_type
    linker_path = os.path.realpath("../src/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    os.system(cmd)
    os.chdir(curr_dir)


def longchat_appy_chat_template(prompt, tokenizer):
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    encoded = tokenizer(prompt)
    return encoded


def mistral_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama2_apply_chat_template(prompt, tokenizer):
    prompt = f"[INST] {prompt} [/INST]"
    encoded = tokenizer(prompt)
    return encoded


def llama3_apply_chat_template(prompt, tokenizer):
    messages = [{"role": "user", "content": f"{prompt}"}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(prompt)
    return encoded


def llama_load_model_and_tokenizer(args, model_name_or_path, **kwargs):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    generate_kwarg = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=True, use_fast=True
    )

    # select apply_chat_template
    if any([x in model_name_or_path.lower() for x in ["llama-2", "llama2", "llama_2"]]):
        print("run llama2 model")
        apply_chat_template = llama2_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        model_type = "llama"

    elif any(
        [
            x in model_name_or_path.lower()
            for x in ["llama-3.1", "llama3.1", "llama_3.1", "llama-3", "llama3", "llama_3"]
        ]
    ):
        print("run llama3 model")
        apply_chat_template = llama3_apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model_type = "llama"

    elif any([
            x in model_name_or_path.lower()
            for x in ["mistral"]
    ]):
        print("run mistral model")
        apply_chat_template = mistral_apply_chat_template
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        model_type = "llama"

    elif any([
        x in model_name_or_path.lower() for x in ["longchat"]
    ]):
        print("run longchat model")
        apply_chat_template = longchat_appy_chat_template
        model_type = "llama"

    else:
        raise ValueError("Unsupported model name")

    set_symlink(model_type, f"modeling_{model_type}_ours.py")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, config=model_config
    )
    if args.skewing_matrix_path is not None:
        A = torch.load(args.skewing_matrix_path)
    for layer in range(len(model.model.layers)):
        model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
        model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
        model.model.layers[layer].self_attn.alpha = args.alpha
        model.model.layers[layer].self_attn.capacity = args.capacity
        model.model.layers[layer].self_attn.budget = args.budget
        if args.skewing_matrix_path is not None:
            model.model.layers[layer].self_attn.skewing_matrix = A[layer]

    # unset to avoid some warning
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.to(torch.float16).eval()

    return model, tokenizer, generate_kwarg, apply_chat_template


def comm_generate(x, generate_kwarg, model, tokenizer):
    input_length = x["input_ids"].shape[1]

    output = model.generate(**x, do_sample=False, **generate_kwarg)

    output = output[:, input_length:]
    preds = tokenizer.batch_decode(output, skip_special_tokens=True)

    return preds
