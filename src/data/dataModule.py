import lightning
from torch.utils.data import DataLoader
from torchvision.datasets import HMDB51

from src.utils import Transform


class DataModule(lightning.LightningDataModule):
    def __init__(self,
                 root: str = "dataset/hmdb51",
                 annotation_path: str = "dataset/splits",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 sharpen: bool = False,
                 gamma: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_wokers = num_workers
        self.train = self._prepare_dataset(train=True)

        # Data augmentation
        if sharpen:
            self.train += self._prepare_dataset(train=True, sharpen=True)

        if gamma:
            self.train = self._prepare_dataset(train=True, gamma=True)

        if sharpen and gamma:
            self.train = self._prepare_dataset(train=True, sharpen=True, gamma=True)

        self.eval = self._prepare_dataset(train=False)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.eval, batch_size=self.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.eval, batch_size=self.batch_size, num_workers=self.hparams.num_workers)

    def _prepare_dataset(self, train: bool = True, sharpen: bool = False, gamma: bool = False) -> HMDB51:
        augmentation_dataset = HMDB51(
            root=self.hparams.root,
            annotation_path=self.hparams.annotation_path,
            frames_per_clip=64,
            train=train,
            transform=Transform(sharpen=sharpen, gamma=gamma),
            fold=1,
            output_format="TCHW"
        )
        return augmentation_dataset
