import html
import os
import time
import torch
import jieba
import re, string
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

# nltk.download('stopwords')
# nltk.download('punkt')

TAG_CLASSES = ["人物", "动物", "时间", "天气", "物品", "地点", "景物"]
TAG_STRING = "、".join(TAG_CLASSES)

# TODO 4.2
# Load prompt generation seq2seq model
promptgen_tokenizer = AutoTokenizer.from_pretrained("./model/promptgen-lexart", trust_remote_code=True)
promptgen_model = AutoModelForCausalLM.from_pretrained("./model/promptgen-lexart", trust_remote_code=True).cuda()
promptgen_model = promptgen_model.eval()
print("promptgen_model loaded")

# TODO 4.3
# Load donbooru tags
synonym_dict = dict()
tag_dict = dict()
danbooru = pd.read_csv('./tags/danbooru.csv')
danbooru.fillna('NaN', inplace=True)
for index, row in danbooru.iterrows():
    if int(row["popularity"]) >= 50:
        tag_dict[row["tag"]] = int(row["popularity"])
        synonym_dict[row["tag"]] = [row["tag"]]
        synonyms = row["synonyms"].split(",")
        for s in synonyms:
            synonym_dict[row["tag"]].append(s.replace("_", " "))
tag_dict = dict(sorted(tag_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))
print("danbooru tags loaded")
# print(synonym_dict)


# TODO 4.4
def enhance_prompts(pos_prompt, tag_dict_):
    pos_prompt = "((masterpiece, best quality, ultra-detailed, illustration)),"  + pos_prompt
    if "1girl" in pos_prompt or "1boy" in pos_prompt:
        pos_prompt += ", ((an extremely delicate and beautiful)), (detailed eyes), (detailed face)"
    neg_prompt = "((nsfw: 1.2)), (EasyNegative:0.8), (badhandv4:0.8), (worst quality, low quality, extra digits), lowres, blurry, text, logo, artist name, watermark"
    return (pos_prompt, neg_prompt)

# TODO 4.2
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

# TODO 4.2
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

# TODO 4.3
def tag_extract(tag_dict_, batch_size=4, mask_ratio=0.3):
    punctuations = [",", ".", "/", ";", "[", "]", "-", "=", "!", "(", ")", "?" "。", "，", "、", "：", "？", "！"]
    words = word_tokenize(",".join([tag_dict_[t] for t in tag_dict_]))
    words = [w for w in words if w not in punctuations]
    words = [PorterStemmer().stem(w) for w in words if w not in set(stopwords.words("english"))]
    # print(words)
    
    def find_tag(word):
        for option in tag_dict:
            for s in synonym_dict[option]:
                if 1.5 * len(word) > len(s) and s.startswith(word):
                    print((word, option, s), end='')
                    return option
        print((word, ), end='')
        return False

    words_ = []
    for w in words:
        tag = find_tag(w)
        if tag:
            words_.append(tag)

    def get_content_word(word):
        word_tags = pos_tag(word_tokenize(word))
        return [w[0] for w in word_tags if w[1][0] in "GMNRV"]

    for t in tag_dict_:
        words_ += get_content_word(tag_dict_[t])
        
    words_ = list(set(words_))
    print(words_)
    
    texts = []
    for i in range(batch_size):
        random_list = sorted(random.sample(range(0, len(words_)), int((1 - mask_ratio) * len(words_))))
        texts.append(", ".join([words_[index] for index in random_list]))
    return [enhance_prompts(t, tag_dict_) for t in texts]
