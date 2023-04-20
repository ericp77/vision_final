import lightning
from torch.utils.data import DataLoader
from torchvision.datasets import HMDB51
from utils.utils import transform


class DataModule(lightning.LightningDataModule):
    def __init__(self,
                 root: str = "dataset/hmdb51",
                 annotation_path: str ="dataset/splits",
                 batch_size: int = 32
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.train = HMDB51(
            root = root,
            annotation_path=annotation_path,
            frames_per_clip=64,
            train=True,
            transform=transform,
            fold=1,
            output_format="TCHW"
        )
        self.eval = HMDB51(
            root=root,
            annotation_path=annotation_path,
            frames_per_clip=64,
            train=False,
            transform=transform,
            fold=1,
            output_format="TCHW"
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eval, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.eval, batch_size=self.batch_size)