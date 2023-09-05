from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from summarization_llm.model import LlamaModule
from summarization_llm.data import SumDataset
import torch


def set_precision():
    torch.set_float32_matmul_precision("medium")

def main():
    set_precision()
    cli = LightningCLI(model_class=LlamaModule, datamodule_class=SumDataset)
    

if __name__ == "__main__":
    main()
