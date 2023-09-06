from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from summarization_llm.model import LlamaModule
from summarization_llm.data import SumDataset
import torch
import transformers
class MyLightningCLI(LightningCLI):
    pass
def set_precision():
    torch.set_float32_matmul_precision("medium")

def main():
    set_precision()
    cli = MyLightningCLI(model_class=LlamaModule,
                         datamodule_class=SumDataset,
                         save_config_kwargs={"overwrite" : True})
    

if __name__ == "__main__":
    main()
