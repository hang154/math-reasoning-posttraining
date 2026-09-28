#!/usr/bin/env python3
import argparse, importlib.machinery, os, sys, types
# The host exposes an incomplete system-wide apex package. Transformers only
# imports ``amp`` eagerly; this run uses native bf16 and never enables Apex.
if "apex" not in sys.modules:
 apex_stub=types.ModuleType("apex"); apex_stub.__spec__=importlib.machinery.ModuleSpec("apex", loader=None); apex_stub.amp=types.SimpleNamespace(); sys.modules["apex"]=apex_stub
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
class Collator:
 def __init__(self,tok,maxlen=768): self.tok=tok; self.max=maxlen
 def __call__(self,batch):
  rows=[]
  for x in batch:
   p=f"Solve the math problem. Show reasoning and end with '#### <number>'.\nProblem: {x['question']}\nAnswer:"
   a=' '+x['answer']; pi=self.tok(p,add_special_tokens=False).input_ids; ai=self.tok(a,add_special_tokens=False).input_ids+[self.tok.eos_token_id]
   ids=(pi+ai)[:self.max]; labels=([-100]*len(pi)+ai)[:self.max]; rows.append((ids,labels))
  L=max(len(x[0]) for x in rows); pad=self.tok.pad_token_id
  ids=[r[0]+[pad]*(L-len(r[0])) for r in rows]; labs=[r[1]+[-100]*(L-len(r[1])) for r in rows]; att=[[1]*len(r[0])+[0]*(L-len(r[0])) for r in rows]
  return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(att),'labels':torch.tensor(labs)}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',default='Qwen/Qwen3-0.6B-Base'); p.add_argument('--out',default='artifacts/posttrain/sft'); p.add_argument('--steps',type=int,default=160); a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True); tok.pad_token=tok.pad_token or tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa'); model.config.use_cache=False
 ds=load_dataset('openai/gsm8k','main',split='train').shuffle(seed=42)
 args=TrainingArguments(output_dir=a.out,max_steps=a.steps,per_device_train_batch_size=8,gradient_accumulation_steps=1,learning_rate=2e-5,warmup_ratio=.03,weight_decay=.01,bf16=True,logging_steps=10,save_strategy='no',report_to='none',remove_unused_columns=False)
 tr=Trainer(model=model,args=args,train_dataset=ds,data_collator=Collator(tok)); tr.train(); tr.save_model(a.out); tok.save_pretrained(a.out)
if __name__=='__main__':main()
