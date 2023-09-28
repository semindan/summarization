from typing import Any, Dict
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from summarization_llm.modelmodule import ModelModule
from summarization_llm.modeling_llama import LlamaForCausalLM
import transformers
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

class LlamaModule(ModelModule):
    def __init__(self, config=None, path='meta-llama/Llama-2-7b-hf', *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = 0

        bnb_4_bit_compute_type = "bfloat16"
        compute_dtype = getattr(torch, bnb_4_bit_compute_type)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            # load_in_8bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=compute_dtype
        )

        model_config = transformers.AutoConfig.from_pretrained(
            path,
        )

        model = LlamaForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=config["r"], 
            lora_alpha=config["lora_alpha"], 
            lora_dropout=config["lora_dropout"], 
            bias=config["bias"], 
            task_type=config["task_type"],
            target_modules = ["q_proj", "v_proj"]
        )
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        dirpath = self.trainer.checkpoint_callback.dirpath
        ckpt_name = f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}"
        self.model.save_pretrained(
                f"{dirpath}/lora-ckpt-{ckpt_name}"
        )
        
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
        # batch["labels"] = torch.where(batch["labels"] != self.tokenizer.pad_token_id, batch["labels"], -100)
        out = self(**batch)
        return self.compute_loss(out.logits, batch["labels"])

    def detokenize(self, predictions):
        return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # batch["labels"] = torch.where(batch["labels"] != self.tokenizer.pad_token_id, batch["labels"], -100)
        out = self(**batch)
        return self.compute_loss(out.logits, batch["labels"])
     
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        print("⚡", "using Llama 2", "⚡")
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # params = self.trainer.model.named_parameters()
        params = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if n in decay_parameters],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in params if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (self.config["adam_beta1"], self.config["adam_beta2"]),
            "eps": self.config["adam_epsilon"],
        }
        optimizer_kwargs["lr"] = self.config["learning_rate"]
        adam_bnb_optim = bnb.optim.PagedAdam32bit(
            optimizer_grouped_parameters,
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
            lr=self.config["learning_rate"],
        )
        return adam_bnb_optim






