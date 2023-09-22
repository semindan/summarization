from typing import Any, Dict
import lightning.pytorch as pl
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from summarization_llm.modelmodule import ModelModule
from summarization_llm.modeling_llama import LlamaForCausalLM

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
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({"pad_token":"<pad>"}) 
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
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
            # device_map={'':torch.cuda.current_device()}
        )

        # should it be there at all then if we pad using </s>?
        # model_config.pad_token_id = self.tokenizer.pad_token_id
        # model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=1)

        lora_config = LoraConfig(
            r=config["r"], 
            lora_alpha=config["lora_alpha"], 
            lora_dropout=config["lora_dropout"], 
            bias=config["bias"], 
            task_type=config["task_type"],
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
        batch["labels"] = torch.where(batch["labels"] != self.tokenizer.pad_token_id, batch["labels"], -100)
        out = self(**batch)
        return out.loss

    def detokenize(self, predictions):
        return self.tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    def generate_predictions(self, logits):
        return torch.argmax(logits, dim=-1)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # output_vanilla = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # breakpoint()

        # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
        # clue: self.model.base_model.model.model.layers[0].self_attn.q_proj.lora_B.default.weight inits to zeros...
        # https://github.com/huggingface/transformers/pull/21955#issuecomment-1454979110 solution?
        # nope...
        # doesn't work with left padding and pad token
        # without left padding and pad token it works (sumdata2)
        # what about pad token only (sumdata3) -- works
        # wtf is it left padding after all? ok...
        # also possible local solutions: https://github.com/huggingface/transformers/issues/25065
        # it works! so just need to prevent underflow, viz. l ~343 in modeling_llama.py

        original_length = batch["input_ids"].size()[1]
        output = self.model.generate(input_ids=batch['input_ids'].squeeze(1),
                                attention_mask=batch['attention_mask'].squeeze(1),
                                # max_length=128,
                                # min_length=original_length + 32,
                                max_new_tokens = 32,
                                # min_new_tokens = 32,
                                no_repeat_ngram_size=3,
                                num_beams=6,
                                # top_p=1,
                                # early_stopping=True
                                # temperature=0.1,
                                # top_p = None,
                                # temperature = None,
                                do_sample=True
                                )
        predictions = self.detokenize(output[:, original_length:]) # include only new tokens
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
        params = self.trainer.model.parameters()
        # params = self.trainer.model.parameters()
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in params if n in decay_parameters],
        #         "weight_decay": self.config["weight_decay"],
        #     },
        #     {
        #         "params": [p for n, p in params if n not in decay_parameters],
        #         "weight_decay": 0.0,
        #     },
        # ]

        # optimizer_kwargs = {
        #     "betas": (self.config["adam_beta1"], self.config["adam_beta2"]),
        #     "eps": self.config["adam_epsilon"],
        # }
        # optimizer_kwargs["lr"] = self.config["learning_rate"]
        # adam_bnb_optim = bnb.optim.Adam8bit(
        #     optimizer_grouped_parameters,
        #     betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
        #     eps=self.config["adam_epsilon"],
        #     lr=self.config["learning_rate"],
        # )
        adam_bnb_optim = bnb.optim.Adam8bit(
            params,
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
            lr=self.config["learning_rate"],
        )
        return adam_bnb_optim



# %%



