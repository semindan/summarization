import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import numpy as np
import datasets
from datasets import dataset_dict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import re
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Any
from pathlib import Path
import os


@dataclass
class SumDataset(pl.LightningDataModule):
    prompt_template: str = ""
    model_path: str = 'meta-llama/Llama-2-13b-chat-hf'
    size: Any = None
    batch_size: int = 2
    seq_length: int = 1024
    rewrite: bool = False
    
    def __post_init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def prepare_data(self):
        mod_path = Path(__file__).parent
        if not os.path.exists(f"{mod_path}/data") or self.rewrite:
            data = load_dataset("xsum")
            data = self.tokenize_data(data)
            data.save_to_disk(f"{mod_path}/data/sumdata.hf")

    def tokenize_data(self, data):
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        def tokenize(example):
            model_inputs = self.tokenizer(
                example["document"],
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
                return_tensors="pt",
            )
            labels = self.tokenizer(
                example["summary"],
                padding="max_length",
                max_length=self.seq_length,
                truncation=True,
                return_tensors="pt",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        data = data.map(tokenize, batched=True)
        return data



    def setup(self, stage: str):
        mod_path = Path(__file__).parent
        data = datasets.load_from_disk(f"{mod_path}/data/sumdata.hf")
        data = data.remove_columns(["document", "summary", "id"])
        print(data["train"].features)
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