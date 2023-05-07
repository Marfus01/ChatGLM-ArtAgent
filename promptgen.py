import html
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


promptgen_tokenizer = AutoTokenizer.from_pretrained("./model/promptgen-lexart", trust_remote_code=True)
promptgen_model = AutoModelForCausalLM.from_pretrained("./model/promptgen-lexart", trust_remote_code=True).cuda()
promptgen_model = promptgen_model.eval()



def enhance_prompts(pos_prompt):
    pos_prompt = "((masterpiece, best quality, ultra-detailed, illustration)),"  + pos_prompt
    neg_prompt = "((nsfw: 1.2)), (EasyNegative:0.8), (badhandv4:0.8), (worst quality, low quality, extra digits), lowres, blurry, text, logo, artist name, watermark"
    return (pos_prompt, neg_prompt)

def generate_batch(input_ids, min_length=20, max_length=300, num_beams=2, temperature=1, repetition_penalty=1, length_penalty=1, sampling_mode="Top K", top_k=12, top_p=0.15):
    top_p = float(top_p) if sampling_mode == 'Top P' else None
    top_k = int(top_k) if sampling_mode == 'Top K' else None

    outputs = promptgen_model.generate(
        input_ids,
        do_sample=True,
        temperature=max(float(temperature), 1e-6),
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        top_p=top_p,
        top_k=top_k,
        num_beams=int(num_beams),
        min_length=min_length,
        max_length=max_length,
        pad_token_id=promptgen_tokenizer.pad_token_id or promptgen_tokenizer.eos_token_id
    )
    texts = promptgen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts


def gen_prompts(text, batch_size=4):
    input_ids = promptgen_tokenizer(text[:256], return_tensors="pt").input_ids
    if input_ids.shape[1] == 0:
        input_ids = torch.asarray([[promptgen_tokenizer.bos_token_id]], dtype=torch.long)
    input_ids = input_ids.to("cuda")
    input_ids = input_ids.repeat((batch_size, 1))

    texts = generate_batch(input_ids)
    print(texts)
    prompt_list = []
    for t in texts:
        prompt_list.append( enhance_prompts(t[0:t.find("Negative")]) )
    return prompt_list


# def add_tab():
#     with gr.Blocks(analytics_enabled=False) as tab:
#         with gr.Row(elem_id="promptgen_main"):
#             with gr.Column(variant="compact"):
#                 with FormRow():
#                     sampling_mode = gr.Radio(label="Sampling mode", elem_id="promptgen_sampling_mode", value="Top K", choices=["Top K", "Top P"])
#                     top_k = gr.Slider(label="Top K", elem_id="promptgen_top_k", value=12, minimum=1, maximum=50, step=1)
#                     top_p = gr.Slider(label="Top P", elem_id="promptgen_top_p", value=0.15, minimum=0, maximum=1, step=0.001)

#                 with gr.Row():
#                     num_beams = gr.Slider(label="Number of beams", elem_id="promptgen_num_beams", value=1, minimum=1, maximum=8, step=1)
#                     temperature = gr.Slider(label="Temperature", elem_id="promptgen_temperature", value=1, minimum=0, maximum=4, step=0.01)
#                     repetition_penalty = gr.Slider(label="Repetition penalty", elem_id="promptgen_repetition_penalty", value=1, minimum=1, maximum=4, step=0.01)

#                 with FormRow():
#                     length_penalty = gr.Slider(label="Length preference", elem_id="promptgen_length_preference", value=1, minimum=-10, maximum=10, step=0.1)
#                     min_length = gr.Slider(label="Min length", elem_id="promptgen_min_length", value=20, minimum=1, maximum=400, step=1)
#                     max_length = gr.Slider(label="Max length", elem_id="promptgen_max_length", value=150, minimum=1, maximum=400, step=1)

#                 with FormRow():
#                     batch_count = gr.Slider(label="Batch count", elem_id="promptgen_batch_count", value=1, minimum=1, maximum=100, step=1)
#                     batch_size = gr.Slider(label="Batch size", elem_id="promptgen_batch_size", value=10, minimum=1, maximum=100, step=1)

#                 with open(os.path.join(base_dir, "explanation.html"), encoding="utf8") as file:
#                     footer = file.read()
#                     gr.HTML(footer)

#         submit.click(
#             fn=ui.wrap_gradio_gpu_call(generate, extra_outputs=['']),
#             _js="submit_promptgen",
#             inputs=[model_selection, model_selection, batch_count, batch_size, prompt, min_length, max_length, num_beams, temperature, repetition_penalty, length_penalty, sampling_mode, top_k, top_p, ],
#             outputs=[res, res_info]
#         )

#     return [(tab, "Promptgen", "promptgen")]

