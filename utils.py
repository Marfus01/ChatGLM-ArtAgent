import os
from typing import Dict, Tuple, Union, Optional
from torch.nn import Module
from transformers import AutoModel
import requests, re, json, io, base64, os
from urllib.parse import quote
import mdtex2html
from bs4 import BeautifulSoup
from PIL import Image, PngImagePlugin
from transformers import AutoModel, AutoTokenizer
import gradio as gr
from promptgen import *
import time
import random


glm_tokenizer = AutoTokenizer.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True)
glm_model = AutoModel.from_pretrained("./model/ChatGLM-6B", trust_remote_code=True).half().quantize(4).cuda()
# glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# glm_model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
glm_model = glm_model.eval()


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in glm_model.stream_chat(glm_tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       


        yield chatbot, history
    print(history)


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

def clear_gallery():
    return [], []


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def translate(word):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    response = requests.post(url, data=key)
    if response.status_code == 200:
        list_trans = response.text
        result = json.loads(list_trans)
        trans = ""
        for r in result['translateResult'][0]:
            trans += r['tgt']
        return trans
    else:
        print(response.status_code)
        return word


def call_sd_t2i(pos_prompt, neg_prompt, width, height, steps, user_input=""):
    url = "http://127.0.0.1:6016"
    payload = {
        "prompt": pos_prompt,
        "steps": steps,
        "negative_prompt": neg_prompt,
        "cfg_scale": 7,
        "n_iter": 2,
        "width": width,
        "height": height,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    # print(r)
    image_list = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)

        # Get Image Info
        # png_payload = {"image": "data:image/png;base64," + i}        
        # response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
        # pnginfo = PngImagePlugin.PngInfo()
        # pnginfo.add_text("parameters", response2.json().get("info"))
        image.save('output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input) + "-" + str(random.randint(1000, 9999)) +'.png')

    return image_list


def call_glm_api(prompt, history, max_length, top_p, temperature):
    url = str(get_config().get('basic').get('host'))+":"+str(get_config().get('basic').get('port'))
    payload = {
        "prompt": prompt,
        "history": history,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature
    }
    response = requests.post(url, json=payload)
    json_resp_raw = response.json()
    json_resp_raw_list = json.dumps(json_resp_raw)
    return json_resp_raw_list



def gen_image_description(user_input, chatbot, max_length, top_p, temperature, history):
    prompt_history = [["我接下来会给你一些作画的指令，你只要回复出作画内容及对象，不需要你作画，不需要给我参考，不需要你给我形容你的作画内容，请直接给出作画内容，你不要不必要的内容，你只需回复作画内容。你听懂了吗","听懂了。请给我一些作画的指令。"]]
    prompt_imput = str(f"我现在要话一副关于“{user_input}”的画，请给出“{user_input}”中的作画内容，请详细描述作画中的内容和对象，并添加一些内容以丰富细节，不要输出多余的信息")
    chatbot.append((parse_text(user_input), ""))
    for response_, history_ in glm_model.stream_chat(glm_tokenizer, prompt_imput, prompt_history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(user_input), parse_text(response_))
    history.append([chatbot[-1][0], chatbot[-1][1]])
    # print(history[-1])
    return chatbot, history



def sd_predict(user_input, chatbot, max_length, top_p, temperature, history, width, height, steps, result_list):
    # Step 1 use ChatGLM-6B associate image description
    # !!! Currently, we don't take history into consideration
    chatbot, history = gen_image_description(user_input, chatbot, max_length, top_p, temperature, history)

    image_description = history[-1][1]
    # image_description = str("").join(image_description.split('\n')[1:])
    # stop_words = ["好的", "我", "将", "会", "画作", "关于", "一张", "画"]
    stop_words = ["\n", "\t", "\r", "<br>"]
    for word in stop_words:
        image_description = image_description.replace(word, "")
    print(image_description)
    image_description = translate(image_description)
    print(image_description)

    # Step 2 use promprGenerater get Prompts
    # prompt_list = gen_prompts(image_description, batch_size=4)
    # print(prompt_list)
    # yield chatbot, history, result_list, []

    # Alternative plan
    prompt_list = [ enhance_prompts(image_description) ] * 4

    # Step 3 use SD get images
    for pos_prompt, neg_prompt in prompt_list:
        new_images = call_sd_t2i(pos_prompt, neg_prompt, width, height, steps, user_input)
        result_list = result_list + new_images
        yield chatbot, history, result_list, new_images
    yield chatbot, history, result_list, result_list



def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


