import pandas as pd 
import numpy as np
import argparse
import shutil
import random
from sklearn.metrics import roc_auc_score
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import os

random.seed(42)
torch.manual_seed(42)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
csv_file = 'SHTech/frame.csv'
def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mempath', type=str, default="checkpoint_epoch_1_3000_5")
    parser.add_argument('--csv_save_path', type=str, default="description")
    parser.add_argument('--test_jump', type=int, default=1)
    args = parser.parse_args()

    mempath = args.mempath
    test_jump = args.test_jump
    csv_save_path = f"{args.csv_save_path}_{os.path.basename(mempath)}.csv"

    #load cogvlm
    model = AutoModelForCausalLM.from_pretrained(
        'cogvlm',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    ).eval()

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

    #setting SSMB 
    mem = torch.load(mempath) 
    mem.mode = 'test'
    mem.memory.first = False
    model.mem = mem

    df = pd.read_csv(csv_file)

    if os.path.exists(csv_save_path):
        done_df = pd.read_csv(csv_save_path, header=None, names=['image_path', 'memloss', 'label', 'answer_org', 'answer_mem'])
        done_set = set(done_df['image_path'].tolist())
    else:
        done_set = set()

    query = 'How many people are in the image and what is each of them doing?'
    results = []

    for idx, row in df.iterrows():
        image_path = row['image_path']
        label = row['label']

        if idx % test_jump != 0:
            continue
        if image_path in done_set:
            print(f"jump: {image_path}")
            continue

        print(f"processing：{image_path}, label: {label}")

        image = Image.open(image_path).convert('RGB')
        inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            memloss = model(
                input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask'],
                images=inputs['images'],
                return_mem_loss=True,
                vision_only=True,
            )[1]

            model.mem = None
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer_org = tokenizer.decode(outputs[0]) 

            model.mem = mem
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            answer_mem = tokenizer.decode(outputs[0]) 

        result = {
            'image_path': image_path,
            'memloss': memloss.detach().item(),
            'label': label,
            'answer_org': answer_org,
            'answer_mem': answer_mem,
        }

        results.append(result)
        pd.DataFrame([result]).to_csv(csv_save_path, mode='a', index=False, header=not os.path.exists(csv_save_path))

    print('Done!')

if __name__ == "__main__":
    main()
