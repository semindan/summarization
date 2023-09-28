from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import datasets
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Any
from pathlib import Path
import os
from summarization_llm.data import DataModule
from itertools import chain


@dataclass
class SumDataset(DataModule):
    prompt: str = ""
    model_path: str = 'meta-llama/Llama-2-7b-hf'
    size: Any = None
    batch_size: int = 2
    seq_length: int = 2048
    overwrite: bool = False
    
    def __post_init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token_id = 0

    def prepare_data(self):
        mod_path = Path(__file__).parent
        if not os.path.exists(f"{mod_path}/data") or self.overwrite:
            data = get_preprocessed_xsum(self.tokenizer)
            data.save_to_disk(f"{mod_path}/data/sumdataconcat.hf")

    def setup(self, stage: str):
        mod_path = Path(__file__).parent
        data = datasets.load_from_disk(f"{mod_path}/data/sumdataconcat.hf")
        data.set_format("pt")
        match stage:
            case "fit":
                self.train = data["train"]
                self.validation = data["validation"]
            case "validation":
                self.validation = data["validation"]
            case "test":
                self.test = data["test"]
            case _:
                raise ValueError("invalid stage")
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    

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



def get_preprocessed_xsum(tokenizer):
    dataset = datasets.load_dataset("xsum")
    prompt = (
        f"Summarize this article:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["document"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset["train"].features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset["train"].features),
    ).map(Concatenator(), batched=True)
    return dataset

