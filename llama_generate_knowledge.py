import argparse
import transformers
import torch
import pandas as pd
import os
import re
import time
from collections import Counter


def parse_arguments():
    parser = argparse.ArgumentParser(description="Knowledge Collection using LLAMA model.")
    parser.add_argument("--input_csv", type=str,default="", help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, default="", help="Path to output CSV file.")
    parser.add_argument("--final_output_csv", type=str, default="", help="Path to final output CSV file.")
    parser.add_argument("--mode", type=str, choices=["percentage", "count", "topk"], default="percentage", help="Mode of phrase filtering.")
    parser.add_argument("--s", type=float, default=0.02, help="Threshold value for filtering.")
    parser.add_argument("--process_proportion", type=float, default=1, help="Proportion of data to process.")
    parser.add_argument("--skip_rule_generate", default = True, action='store_true', help="Skip rule generation if set.")
    parser.add_argument("--model_id", type=str, default="LLAMA3.1", help="Model ID for LLAMA model.")
    return parser.parse_args()

def get_rule(summary_text, pipeline):
    prompt = ("What is people doing in the following description? Summarize each different type of human activities "
              "using a single verb or simple verb-noun phrase. Only verb and noun are allowed. "
              "(like 'walking, using phone').\n"
              f"Description: {summary_text}\n"
              "Provide your answer in the following format: \n"
              "Activities: walking, using phone...")
    
    messages = [
        {"role": "system", "content": "You are a video surveillance analyst."},
        {"role": "user", "content": prompt}
    ]
    
    outputs = pipeline(messages, max_new_tokens=256, do_sample=False)
    response = outputs[0]['generated_text'][0]['content']
    
    activities_start = response.find("Activities: ") + len("Activities: ")
    activities = response[activities_start:].split("\n")[0].strip()
    return activities if activities else "None"

def refine_rules(args, pipeline):
    if pipeline == None:
        pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",)
        
    with open(args.final_output_csv, "r") as f:
        lines = f.readlines()
    if not lines:
        return
    original_rules = lines[0].strip()
    if not original_rules:
        return
    
    prompt = ("Here is a list of human activities list. Some of them may have structural repetition or phrase pattern repetition. Please simplify the list.\n"
              "Provide your response in the following format:"
              "New List:...,...,....(Use ',' to separate)."         
              f"Now the list is: {original_rules}\n")
    
    messages = [{"role": "system", "content": "You are a language expert."},
                {"role": "user", "content": prompt}]
    
    outputs = pipeline(messages, max_new_tokens=256, do_sample=False)
    response = outputs[0]['generated_text'][2]['content']
    print(outputs[0]['generated_text'][2]['content'])
    
    rule_start = response.find("New List:") + len("New List:")
    #rules = response[rule_start:].split("\n")[0].strip()
    rules = response[rule_start:].strip()
    
    refined_output_csv = args.final_output_csv.replace(".csv", "_refined.csv")
    with open(refined_output_csv, "w") as f:
        f.write(rules + "\n")
    print(f"Refined rules saved to {refined_output_csv}.")

def process_data(args, pipeline): 
    df = pd.read_csv(args.input_csv)
    output_data = []
    for _, row in df.iterrows():
        image_path, memloss, label, answer_mem = row['image_path'], row['memloss'], row['label'], row['answer_mem']
        if label == 0:
            print(f"Processing image: {image_path}")
            try:
                new_summary = get_rule(answer_mem, pipeline)
                output_data.append({'image_path': image_path, 'memloss': memloss, 'label': label, 'summary': new_summary})
                pd.DataFrame(output_data).to_csv(args.output_csv, index=False)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def filter_phrases(args):
    existing_results = pd.read_csv(args.output_csv)
    sampled_results = existing_results.sample(frac=args.process_proportion, random_state=42).sort_index()
    summaries = sampled_results["summary"].dropna()

    phrases = [
        re.sub(r"[^\w\s]", "", phrase).strip().lower()
        for summary in summaries
        for phrase in summary.split(",")
    ]

    phrase_counts = Counter(phrases)
    sorted_phrases = phrase_counts.most_common()
    
    if args.mode == 'percentage':
        filtered_phrases = [phrase for phrase, count in sorted_phrases if count / sum(phrase_counts.values()) > args.s]
    elif args.mode == 'count':
        filtered_phrases = [phrase for phrase, count in sorted_phrases if count > args.s]
    elif args.mode == 'topk':
        filtered_phrases = [phrase for phrase, _ in sorted_phrases[:int(args.s)]]
    else:
        raise ValueError("Invalid mode. Choose 'percentage', 'count', or 'topk'.")
    
    with open(args.final_output_csv, "w") as f:
        f.write(",".join(filtered_phrases) + "\n")
        f.write(",".join(map(str, [phrase_counts[phrase] for phrase in filtered_phrases])))
    print(f"Filtered phrases saved to {args.final_output_csv}")

def main():
    args = parse_arguments()   
    pipeline = None
    if not args.skip_rule_generate:
        pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        process_data(args, pipeline)
    filter_phrases(args)
    refine_rules(args, pipeline)  # Uncomment to refine rules
    
if __name__ == "__main__":
    main()