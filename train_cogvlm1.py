import argparse
import gc
import json
import os
import random
import threading

import yaml
from PIL import Image
import psutil
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    LlamaTokenizer,
    BitsAndBytesConfig
)
from torch.utils.tensorboard import SummaryWriter

from peft import get_peft_model, LoraConfig, TaskType
from model.memory_module import MemModule

import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024
                 ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir,
                                      'labels_en')  # can be change to labels_en or labels_zh in SFT-311K dataset
        self.filenames = os.listdir(self.image_dir)
        self.input_length = input_length
        self.output_length = output_length
        self.device = device
        self.torch_type = torch_type
        self.padding_len = 1224
        self.max_length = self.input_length + self.output_length + self.padding_len

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        label_name = os.path.join(self.label_dir, self.filenames[idx].replace('.jpg', '.json'))

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label_data = json.load(f)

        num_rounds = len(label_data["conversations"]) // 2
        sampled_round_id = random.randint(0, num_rounds - 1)
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None
        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=history,
            images=[image],
            answer=response
            #answer = None,
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full([pad_to_length - current_length],
                            fill_value=pad_value,
                            dtype=unpadded_tensor.dtype,
                            device=unpadded_tensor.device)), dim=0)

        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.max_length,
            pad_value=0,
        )

        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.max_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            self.max_length,
            pad_value=0
        )

        input_data['labels'] = pad_to_len(
            input_data['labels'],
            self.max_length,
            pad_value=-100
        )

        for data_key in input_data:
            if data_key in ['images']:
                input_data[data_key] = [data.to(self.device).to(self.torch_type) for data in
                                        input_data[data_key]]
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        return input_data

def b2mb(x):
    return int(x / 2 ** 20)

class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

def main():
    parser = argparse.ArgumentParser(description="Finetune a CogVLM model with LoRA")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--torch_type", type=str, default="torch.bfloat16", help="Torch type")
    parser.add_argument("--print_step", type=int, default=10, help="Steps between checkpoints")
    parser.add_argument("--save_step", type=int, default=1, help="Steps between checkpoints")
    parser.add_argument("--train_dataset_rate", type=float, default=1,
                        help="Proportion of dataset to use for training")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=128, help="Maximum output length")
    parser.add_argument("--model_path", type=str,
                        default='cogvlm',
                        help="Path to the pretrained model")
    parser.add_argument("--dataset_path", type=str,
                        default="dataset/shanghai_mini",
                        help="Path to the conversation dataset")
    parser.add_argument("--save_path", type=str, default="save_shanghai",
                        help="Path to save the finetuned model, must be a exit directory")
    parser.add_argument("--ds_config", type=str, default="cogvlm2/finetune_demo/ds_config.yaml",
                        help="DeepSpeed configuration file path")
    parser.add_argument('--mem_dim', type=int, default=3000, help='memory dimention')
    parser.add_argument('--topk', type=int, default=5, help='memory dimention')
    parser.add_argument('--text_loss_weight', type=int, default=1, help='memory loss weight')
    

    args = parser.parse_args()
    args.torch_type = eval(args.torch_type)

    save_path = f"{args.save_path}_{args.mem_dim}_{args.topk}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(args.ds_config) as f:
        ds_config = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_config)

    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    #device_map='auto',
    device_map='balanced_low_0',
    torch_dtype=args.torch_type,
    trust_remote_code=True,
    #quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    low_cpu_mem_usage=True
    )

    mem = MemModule(mem_dim=args.mem_dim, fea_dim=4096, topk=args.topk).to(torch.bfloat16)
    model.mem = mem
    print(model.mem)

    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))
    dataset = ConversationDataset(
        root_dir=args.dataset_path,
        tokenizer=tokenizer,
        model=model,
        torch_type=args.torch_type,
        input_length=args.max_input_len,
        output_length=args.max_output_len
    )
    train_size = int(args.train_dataset_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,

    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )

    #froze all other block
    for name, param in model.named_parameters():
        if 'mem' in name:
            print(name)
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.mem.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )


    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    logger.info("Preparation done. Starting training...")
    writer = SummaryWriter(log_dir=args.save_path)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            out = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels'],
                return_mem_loss = True,
            )
            outputs = out[0]
            memloss = out[1]
            loss =  memloss + outputs.loss*args.text_loss_weight
            total_loss += memloss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if (step + 1) % args.print_step == 0:
                print(f"Epoch {epoch}, Step {step + 1}, Loss {outputs.loss.item()}")
                print(f"Epoch {epoch}, Step {step + 1}, Lossmem {memloss.detach().item()}")
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_dataloader) + step)

        checkpoint_path = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}_{args.mem_dim}_{args.topk}')
        if (epoch + 1) % args.save_step == 0:
            torch.save(model.mem, checkpoint_path)

        total_loss = accelerator.gather(total_loss)
        avg_loss = total_loss.mean().item() / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor(avg_loss))
        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Perplexity', train_ppl, epoch)
        accelerator.print(f"Epoch {epoch}: Average Loss {avg_loss:.4f}, Perplexity {train_ppl:.4f}")

if __name__ == "__main__":
    main()
