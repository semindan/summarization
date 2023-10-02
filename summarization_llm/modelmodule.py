from typing import Any
import lightning.pytorch as pl
# from torchmetrics.text.rouge import ROUGEScore
class ModelModule(pl.LightningModule):
    def __init__(self, config=None, path=None,  *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.rouge = ROUGEScore()
        self.validation_outputs = []
    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        self.log("train/train_loss",
                outputs["loss"],
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)

        return super().on_train_batch_end(outputs, batch, batch_idx)
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.log("validation/validation_loss",
                outputs,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True)
        # self.validation_outputs.append(outputs)
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self) -> None:
        pass
        # for pred, ref in self.validation_outputs:
        #     print(pred, ref)
        #     self.rouge(pred, ref)
        # rouge_type_results = self.rouge.compute()
        # for metric, result in rouge_type_results.items():
        #     self.log(metric, result, on_epoch=True, logger=True, sync_dist=True)
        # self.validation_outputs.clear()
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass


    def configure_optimizers(self):
        pass
