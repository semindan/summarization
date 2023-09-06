import lightning.pytorch as pl
import torch
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss


from torch import cuda, bfloat16
import transformers

import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

from torchmetrics.text.rouge import ROUGEScore

class LlamaModule(pl.LightningModule):
    def __init__(self, config=None, path='meta-llama/Llama-2-13b-chat-hf', *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.rouge = ROUGEScore()
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

        model_config = transformers.AutoConfig.from_pretrained(
            path,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
        )

        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
        self.validation_outputs = [[],[]]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        reference_ids=None,
        *args,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=reference_ids,
        )

    def compute_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        labels = batch["labels"].to(out.logits.device)
        out_loss = self.compute_loss(out.logits, labels)
        print(out_loss)
        self.log(
                "train_loss",
                out_loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        return out_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        self.validation_outputs[0].append(self.tokenizer.batch_decode(predictions, skip_special_tokens=True))
        self.validation_outputs[1].append(self.tokenizer.batch_decode(references, skip_special_tokens=True))
        return predictions, references

    def on_validation_epoch_end(self) -> None:
        all_preds = self.validation_outputs[0]
        all_refs = self.validation_outputs[1]
        for pred, ref in zip(all_preds, all_refs):
            self.rouge(pred, ref)
        results = self.rouge.compute()
        for metric, result in results.items():
            self.log(metric, result, on_epoch=True, logger=True)
        self.validation_outputs.clear()
        self.validation_outputs = [[], []]
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        return predictions, references

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        return predictions, references

    def configure_optimizers(self):
        print("⚡", "using Llama 2", "⚡")
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (self.config["adam_beta1"], self.config["adam_beta2"]),
            "eps": self.config["adam_epsilon"],
        }
        optimizer_kwargs["lr"] = self.config["learning_rate"]
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
            lr=self.config["learning_rate"],
        )
        return adam_bnb_optim



# %%



