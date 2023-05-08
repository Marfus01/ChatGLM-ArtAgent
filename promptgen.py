import html
import os
import time
import torch
import jieba
import nltk
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load prompt generation seq2seq model
promptgen_tokenizer = AutoTokenizer.from_pretrained("./model/promptgen-lexart", trust_remote_code=True)
promptgen_model = AutoModelForCausalLM.from_pretrained("./model/promptgen-lexart", trust_remote_code=True).cuda()
promptgen_model = promptgen_model.eval()

# Load donbooru tags
synonym_dict = []
tag_dict = []
danbooru = pd.read_csv('./tags/danbooru.csv')
danbooru.fillna('NaN', inplace=True)
for index, row in danbooru.iterrows():
    tag_dict[row["tag"]] = int(row["popularity"])
    synonym_dict[row["tag"]] = row["tag"]
    synonyms = row["synonyms"]
    for s in synonyms:
        synonym_dict[s] = row["tag"]


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

def tag_extract(text, batch_size=4, mask_ratio=0.2):




    for t in texts:
        prompt_list.append( enhance_prompts(t[0:t.find("Negative")]) )
    return prompt_list

