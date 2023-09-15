import lightning.pytorch as pl
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from summarization_llm.modelmodule import ModelModule

from torch import cuda, bfloat16
import transformers

import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

from torchmetrics.text.rouge import ROUGEScore

class LlamaModule(ModelModule):
    def __init__(self, config=None, path='meta-llama/Llama-2-13b-chat-hf', *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(path)
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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        *args,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def compute_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        labels = batch["labels"].to(out.logits.device)
        out_loss = self.compute_loss(out.logits, labels)
        print(out_loss, out.loss)
        
        return out_loss

    def detokenize(self, predictions):
        return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    def generate_predictions(self, logits):
        return torch.argmax(logits, dim=-1)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predictions = self.detokenize(self.generate_predictions(out.logits))
        references = self.detokenize(batch["labels"])
        return predictions, references
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predictions = self.detokenize(self.generate_predictions(out.logits))
        return predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        predictions = self.detokenize(self.generate_predictions(out.logits))
        return predictions

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



