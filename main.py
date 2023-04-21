import argparse
import os.path

import torch
from lightning import Trainer

from data.dataModule import DataModule
from model.wrapperModel import WrapperModel

from lightning.pytorch.loggers import WandbLogger

def get_model_args():
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("--lr", type=float, default=3e-5)

    return model_parser.parse_args()

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

def get_config_args():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--wandb", type=bool, default=False)
    config_parser.add_argument("--log_dir", type=str, default="logs")

    return config_parser.parse_args()


if __name__ == "__main__":
    model_args = get_model_args()
    trainer_args = get_trainer_args()
    data_args = get_data_args()
    config_args = get_config_args()

    logger = None
    if config_args.wandb:
        if not os.path.exists(config_args.log_dir):
            os.makedirs(config_args.log_dir)

        logger = WandbLogger(
            project="vision_group",
            name=f"base_{model_args.lr}_{data_args.batch_size}",
            save_dir=config_args.log_dir
        )

    data_module = DataModule(**vars(data_args))

    wrapper_model = WrapperModel(
        num_class=len(data_module.train.classes),
        lr=model_args.lr,
    )
    wrapper_model.model = torch.compile(wrapper_model.model)

    trainer = Trainer(**vars(trainer_args))
    trainer.fit(model=wrapper_model,
                datamodule=data_module)
