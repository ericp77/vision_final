import argparse

import torch
from lightning import Trainer

from data.dataModule import DataModule
from model.wrapperModel import WrapperModel

def get_trainer_args():
    trainer_parser = argparse.ArgumentParser()
    trainer_parser.add_argument("--devices", type=int, default=1)
    trainer_parser.add_argument("--accelerator", type=str, default="cpu")
    trainer_parser.add_argument("--max_epochs", type=int, default=50)

    return trainer_parser.parse_args()

def get_data_args():
    data_parser = argparse.ArgumentParser()
    data_parser.add_argument("--root", type=str, default="dataset/hmdb51")
    data_parser.add_argument("--annotation_path", type=str, default="dataset/splits")
    data_parser.add_argument("--batch_size", type=int, default=8)

    return data_parser.parse_args()


if __name__ == "__main__":
    trainer_args = get_trainer_args()
    data_args = get_data_args()

    data_module = DataModule(**vars(data_args))

    wrapper_model = WrapperModel(num_class=len(data_module.train.classes))
    wrapper_model.model = torch.compile(wrapper_model.model)

    trainer = Trainer(**vars(trainer_args))
    trainer.fit(model=wrapper_model,
                datamodule=data_module)
