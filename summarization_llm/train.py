from lightning.pytorch.cli import LightningCLI
from summarization_llm.modelmodule import ModelModule
from summarization_llm.data import DataModule
import torch
from lightning import Trainer

def set_precision():
    torch.set_float32_matmul_precision("medium")

def main():
    set_precision()
    cli = LightningCLI(model_class=ModelModule,
                         trainer_class=Trainer,
                         datamodule_class=DataModule,
                         save_config_kwargs={"overwrite" : True},
                         subclass_mode_model=True,
                         subclass_mode_data=True,)
    
if __name__ == "__main__":
    main()
