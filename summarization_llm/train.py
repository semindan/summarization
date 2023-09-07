from lightning.pytorch.cli import LightningCLI
from summarization_llm.modelmodule import ModelModule
from summarization_llm.data import SumDataset
import torch
import transformers
class MyLightningCLI(LightningCLI):
    pass
def set_precision():
    torch.set_float32_matmul_precision("medium")

def main():
    set_precision()
    cli = MyLightningCLI(model_class=ModelModule,
                         datamodule_class=SumDataset,
                         save_config_kwargs={"overwrite" : True},
                         subclass_mode_model=True)
    

if __name__ == "__main__":
    main()
