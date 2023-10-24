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
from accelerate import infer_auto_device_map

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
            # load_in_4bit=True,
            load_in_8bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=compute_dtype
        )

        model_config = transformers.AutoConfig.from_pretrained(
            path,
        )

        # breakpoint()
        model = LlamaForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
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
        inputs_embeds=None,
        *args,
        **kwargs
    ):
        # if inputs_embeds is None:
        #     inputs_embeds = self.model.base_model.model.model.embed_tokens(input_ids)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            # inputs_embeds=inputs_embeds
        )

    def training_step(self, batch, batch_idx):
        # batch["labels"] = torch.where(batch["labels"] != self.tokenizer.pad_token_id, batch["labels"], -100)
        out = self(**batch)
        return out.loss

    def detokenize(self, predictions):
        return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # batch["labels"] = torch.where(batch["labels"] != self.tokenizer.pad_token_id, batch["labels"], -100)
        out = self(**batch)
        return out.loss
    def on_validation_epoch_end(self) -> None:
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
        model_input = self.tokenizer(eval_prompt, return_tensors="pt")
        with torch.no_grad():
            print(self.tokenizer.decode(self.model.generate(**model_input, max_new_tokens=100, top_k=3, do_sample=True)[0], skip_special_tokens=True).replace(eval_prompt,"- "))
        
     
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        print("⚡", "using Llama 2", "⚡")
        # decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        # decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # params = self.trainer.model.named_parameters()
        # params = self.model.named_parameters()
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
        # adam_bnb_optim = bnb.optim.AdamW(
        #     optimizer_grouped_parameters,
        #     betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
        #     eps=self.config["adam_epsilon"],
        #     lr=self.config["learning_rate"],
        # )
        params = self.trainer.model.parameters()
        adam_bnb_optim = bnb.optim.AdamW8bit(
            params,
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            eps=self.config["adam_epsilon"],
            lr=self.config["learning_rate"],
        )
        return adam_bnb_optim






