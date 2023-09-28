# %% [markdown]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# %% [markdown]
# ## Quick Start Notebook
# 
# This notebook shows how to train a Llama 2 model on a single GPU (e.g. A10 with 24GB) using int8 quantization and LoRA.
# 
# ### Step 0: Install pre-requirements and convert checkpoint
# 
# The example uses the Hugging Face trainer and model which means that the checkpoint has to be converted from its original format into the dedicated Hugging Face format.
# The conversion can be achieved by running the `convert_llama_weights_to_hf.py` script provided with the transformer package.
# Given that the original checkpoint resides under `models/7B` we can install all requirements and convert the checkpoint with:

# %% [markdown]
# ### Step 1: Load the model
# 
# Point model_id to model weight folder

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.cuda.get_device_name(0))

# %%
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_id = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_id)

model = LlamaForCausalLM.from_pretrained(
    model_id, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
)


# %% [markdown]
# ### Step 2: Load the preprocessed dataset
# 
# We load and preprocess the samsum dataset which consists of curated pairs of dialogs and their summarization:

# %%
from dataclasses import dataclass

@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048

# %%
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


# %%
import datasets

def get_preprocessed_xlsum(dataset_config, tokenizer, split="train"):
    dataset = datasets.load_dataset("csebuetnlp/xlsum","english", split=split)

    prompt = (
        f"Summarize this article:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["text"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset

# %%
dataset = datasets.load_dataset("csebuetnlp/xlsum", "english", split="train")
prompt = f"Summarize this article:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"

def apply_prompt_template(sample):
    return {
        "text": prompt.format(
            dialog=sample["text"],
            summary=sample["summary"],
            eos_token=tokenizer.eos_token,
        )
    }


# dataset_promptified = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))


# # %%
# dataset_promptified[:3]

# %%
from pathlib import Path
import os
import sys


# train_dataset = get_preprocessed_xlsum(samsum_dataset, tokenizer, 'train')
# validation_dataset = get_preprocessed_xlsum(samsum_dataset, tokenizer, 'validation')
# train_dataset.save_to_disk("/home/semindan/summarization_llm_pipeline/summarization_llm/data/xlsum_train_base.hf")
# validation_dataset.save_to_disk("/home/semindan/summarization_llm_pipeline/summarization_llm/data/xlsum_validation_base.hf")

train_dataset = datasets.load_from_disk("/home/semindan/summarization_llm_pipeline/summarization_llm/data/xlsum_train_base.hf")
validation_dataset = datasets.load_from_disk("/home/semindan/summarization_llm_pipeline/summarization_llm/data/xlsum_validation_base.hf")
# # %% [markdown]
# # 

# # %%
# import pickle
# with open("/home/ullriher/ullriher/xlsum_train.pkl","rb") as f:
#     train_dataset = pickle.load(f)

# %%
# train_dataset[:3]

# %% [markdown]
# ### Step 3: Check base model
# 
# Run the base model on an example input:

# %%
eval_prompt = """
Summarize this article:
Hollywood actor Kevin Spacey wept in court as he was cleared of all charges in his sexual assault trial in London.
Jurors at Southwark Crown Court returned not guilty verdicts for nine sexual offence charges relating to four men between 2001 and 2013.
Speaking afterwards, Mr Spacey said he was "grateful" to the jury as he thanked them for their deliberations - which took more than 12 hours.
Outside the court the Oscar winner added he was "humbled".
The US actor was acquitted of seven counts of sexual assault, one count of causing a person to engage in sexual activity without consent and one count of causing a person to engage in penetrative sexual activity without consent.
After the verdict was read, he put his hand on his chest, looked at the jurors and mouthed "thank you" twice before they left the room.
Addressing journalists on the court's steps Mr Spacey said there was "a lot for me to process".
"I would like to say that I am enormously grateful to the jury for having taken the time to examine all of the evidence and all of the facts carefully before they reached their decision," he said.
"I am humbled by the outcome today. I also want to thank the staff inside this courthouse, the security, and all of those who took care of us every single day."
Kevin Spacey: Who is the Oscar-winning actor?
Jurors rejected the prosecution's claims Mr Spacey had "aggressively" grabbed three men by the crotch and had performed a sex act on another man while he was asleep in his flat.

---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# %% [markdown]
# We can see that the base model only repeats the conversation.
# 
# ### Step 4: Prepare model for PEFT
# 
# Let's prepare the model for Parameter Efficient Fine Tuning (PEFT):

# %%
# !pip install peft

# %%
import torch
torch.__version__

# %%
model.train()

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)



# %%
import transformers
transformers.__version__

# %% [markdown]
# ### Step 5: Define an optional profiler

# %%


# %%
from transformers import TrainerCallback
from contextlib import nullcontext
import wandb 
bs = 4
enable_profiler = False
output_dir = f"/home/semindan/summarization_llm_pipeline/summarization_llm/checkpoints/bertik_llama"

config = {
    'lora_config': lora_config,
    'learning_rate': 5e-5,
    'num_train_epochs': .14,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': bs,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()

# %%
train_dataset, train_dataset.select(range(50000))

# %% [markdown]
# ### Step 6: Fine tune the model
# 
# Here, we fine tune the model for a single epoch which takes a bit more than an hour on a A100.

# %%
from transformers import default_data_collator, Trainer, TrainingArguments

# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=2,
    save_strategy="no",
    optim="adamw_hf",
    report_to="wandb",
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.select(range(50000)),
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()

# %%
# clear cuda mem
torch.cuda.empty_cache()

# %%
model.save_pretrained(output_dir)


# %%
eval_prompt = """
Summarize this article:
Niger's President Mohamed Bazoum has issued a defiant message on Twitter after soldiers announced a coup overnight in the West African nation.
Trouble began early on Wednesday when troops from the presidential guard took him captive.
His foreign minister has said the takeover does not have the backing of the whole military, but the army chief has now said he backs the junta.
Mr Bazoum is a key Western ally in the fight against Islamist militants.
The US and France both have military bases in the uranium-rich country - and have condemned the coup.
US Secretary of State Antony Blinken called up Mr Bazoum promising Washington's "unwavering support" and the UN and the European Union have called for the president's immediate release.
Africa Live: Updates on this and other stories from the continent
Putin's show: Which African leaders will have star role?
Are military takeovers on the rise in Africa?
The 64-year-old, who was elected Niger's president two years ago, took to Twitter on Thursday morning to say: "The hard-won achievements will be safeguarded. All Nigeriens who love democracy and freedom will see to it."
The capital, Niamey, is currently deserted, but this is largely because it has been raining heavily all morning.
Even a march planned by those who support the takeover has not happened because of the downpours.
But people in Niger are sharply divided about the turn of events.
Some are shocked and upset and while it was under way on Wednesday, hundreds of the president's supporters defied the soldiers to go out on to the streets and call for the military to return to the barracks.
They dispersed after warning shots were fired - the only gunfire heard in this bloodless seizure of power.
They have said they will not accept the coup but it is not clear how they will oppose it. They have not called any more streets protests for the time-being.

---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(eval_prompt.replace("Summary:", "Claims:"))
    for i in range(5):
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100, top_k=3, do_sample=True)[0], skip_special_tokens=True).replace(eval_prompt,"- "))


# # %%
# model.generate(**model_input, max_new_tokens=100, top_k=3, do_sample=True)

# # %%
# model_input['input_ids'].shape

# # %%
# model.config

# # %%
# train_dataset['input_ids']

# # %%
# len(train_dataset['input_ids'][0])

# # %%
# train_data = [len(train_dataset['input_ids'][i]) for i in range(len(train_dataset['input_ids']))]

# # %%
# train_dataset['input_ids'][:2]

# # %%
# train_data

# # %%
# dataset = datasets.load_dataset("csebuetnlp/xlsum","english", split="train")

# prompt = (
#     f"Summarize this article:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
# )
# def apply_prompt_template(sample):
#     return {
#         "text": prompt.format(
#             dialog=sample["text"],
#             summary=sample["summary"],
#             eos_token=tokenizer.eos_token,
#         )
#     }
# dataset_prompts = dataset.select(range(30)).map(apply_prompt_template, remove_columns=list(dataset.features))
    
# dataset_tokens = dataset_prompts.map(
#     lambda sample: tokenizer(sample["text"]),
#     batched=True,
#     remove_columns=list(dataset_prompts.features),
# ).map(Concatenator(), batched=True)

# # dataset_detokens = dataset_tokens.map(lambda sample: tokenizer.decode(sample["input_ids"]), batched=True)

# # %%
# print(dataset_prompts[1]['text'])

# # %%
# len(dataset_prompts)

# # %%
# len(dataset_tokens)

# # %%
# dataset_detokens = [tokenizer.decode(dataset_tokens["input_ids"][i], batched=True) for i in range(len(dataset_tokens))]

# # %%
# len(dataset_prompts), len(dataset_tokens)

# # %%
# [tokenizer.decode(dataset_tokens["input_ids"][i], batched=True) for i in range(len(dataset_tokens))]

# # %%



