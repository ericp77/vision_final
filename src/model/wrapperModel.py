from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

from src.model.model import S3D
from src.utils.utils import load_parameters, get_class_name


class WrapperModel(pl.LightningModule):
    def __init__(self, num_class: int, lr: float = 1e-03):
        super().__init__()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_class)
        self.top_5_train_accuracy = Accuracy(task="multiclass", num_classes=num_class, top_k=5)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_class)
        self.top_5_val_accuracy = Accuracy(task="multiclass", num_classes=num_class, top_k=5)

        # to calculate accuracy for each class in validation
        self.accuracies = [Accuracy(task="multiclass", num_classes=num_class) for _ in range(num_class)]
        self.all_preds = []
        self.all_labels = []

        self.model = S3D(num_class=num_class)

        load_parameters('./model/S3D_kinetics400.pt', self.model)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # move accuracy func to device
        self.train_accuracy = self.train_accuracy.to(self.device)
        self.top_5_train_accuracy = self.top_5_train_accuracy.to(self.device)

        # log train loss
        self.log('train_loss', loss)
        self.log('train_acc', self.train_accuracy(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc_top_5', self.top_5_train_accuracy(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # move accuracy func to device
        self.val_accuracy = self.val_accuracy.to(self.device)
        self.top_5_val_accuracy = self.top_5_val_accuracy.to(self.device)

        # log val loss
        self.log('val_loss', loss)
        self.log('val_acc', self.val_accuracy(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc_top_5', self.top_5_val_accuracy(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)

        # to calculate accuracy for each class in validation
        self.all_preds.append(torch.softmax(y_hat, dim=1))
        self.all_labels.append(y)

        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.all_preds, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)

        acc_per_class = []
        for i in range(self.hparams.num_class):
            class_mask = (all_labels == i)
            class_preds = all_preds[class_mask]
            class_labels = all_labels[class_mask]

            # move accuracy func to device
            self.accuracies[i] = self.accuracies[i].to(self.device)

            if class_preds.shape[0] != 0:
                acc_per_class.append(self.accuracies[i](class_preds, class_labels))

        self.log_dict({f"val_acc_{get_class_name(i)}": acc.item() for i, acc in enumerate(acc_per_class)})
        self.all_preds = []
        self.all_labels = []

    def configure_optimizers(self) -> Any:
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optim

