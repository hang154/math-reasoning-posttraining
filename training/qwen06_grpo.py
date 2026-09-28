#!/usr/bin/env python3
import argparse,importlib.machinery,re,sys,types
if "apex" not in sys.modules:
 apex_stub=types.ModuleType("apex"); apex_stub.__spec__=importlib.machinery.ModuleSpec("apex", loader=None); apex_stub.amp=types.SimpleNamespace(); sys.modules["apex"]=apex_stub
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

def gold_num(s):
 m=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',s or ''); return m[-1].replace(',','') if m else None
def pred_num(s):
 m=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',s or '');
 if m:return m[-1].replace(',','')
 m=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',s or ''); return m[-1].replace(',','') if m else None
def reward(completions,answer,**kwargs): return [1.0 if pred_num(c)==gold_num(g) else 0.0 for c,g in zip(completions,answer)]
def format_reward(completions,**kwargs): return [0.2 if re.search(r'####\s*[-+]?\d',c or '') else 0.0 for c in completions]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',default='artifacts/posttrain/sft'); p.add_argument('--out',default='artifacts/posttrain/grpo'); p.add_argument('--steps',type=int,default=60); a=p.parse_args()
 ds=load_dataset('openai/gsm8k','main',split='train').shuffle(seed=123).select(range(3000))
 ds=ds.map(lambda x:{'prompt':f"Solve the problem and end with #### <number>.\n{x['question']}\n",'answer':x['answer']},remove_columns=[c for c in ds.column_names if c not in ['answer']])
 cfg=GRPOConfig(output_dir=a.out,max_steps=a.steps,learning_rate=5e-6,per_device_train_batch_size=2,gradient_accumulation_steps=1,num_generations=4,max_prompt_length=512,max_completion_length=256,bf16=True,logging_steps=1,save_strategy='no',report_to='none')
 tr=GRPOTrainer(model=a.model,reward_funcs=[reward,format_reward],args=cfg,train_dataset=ds); tr.train(); tr.save_model(a.out)
if __name__=='__main__':main()
