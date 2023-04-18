import argparse

import torch
from lightning import Trainer

from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader

from dataModule import DataModule
from wrapperModel import WrapperModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)

    data_module = DataModule(
        root="dataset/hmdb51",
        annotation_path="dataset/splits",
    )

    model = WrapperModel(num_class=len(data_module.train.classes))

    trainer = Trainer(**parser.parse_args())
    trainer.fit(model=model,
                datamodule=data_module)
