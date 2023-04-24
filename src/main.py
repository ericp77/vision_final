import argparse
import os.path
from pathlib import Path

import torch
from lightning import Trainer

from src.data.dataModule import DataModule
from src.model.wrapperModel import WrapperModel

from lightning.pytorch.loggers import WandbLogger


ROOT_PATH = Path(os.path.abspath(__file__)).parent.parent
def get_model_args():
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("--lr", type=float, default=1e-5)
    return model_parser.parse_args()

def get_trainer_args():
    trainer_parser = argparse.ArgumentParser()
    trainer_parser.add_argument("--devices", type=int, default=-1)
    trainer_parser.add_argument("--accelerator", type=str, default="auto")
    trainer_parser.add_argument("--max_epochs", type=int, default=10)
    trainer_parser.add_argument("--precision", type=str, default="16-mixed")

    return trainer_parser.parse_args()

def get_data_args():
    data_parser = argparse.ArgumentParser()
    dataset_path = Path(ROOT_PATH, "dataset/hmdb51")
    annotation_path = Path(ROOT_PATH, "dataset/splits")

    data_parser.add_argument("--root", type=str, default=str(dataset_path))
    data_parser.add_argument("--annotation_path", type=str, default=str(annotation_path))
    data_parser.add_argument("--batch_size", type=int, default=12)
    data_parser.add_argument("--num_workers", type=int, default=4)
    data_parser.add_argument("--sharpen", type=bool, default=False)
    data_parser.add_argument("--gamma", type=bool, default=False)

    return data_parser.parse_args()

def get_config_args():
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--wandb", type=bool, default=True)
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

    trainer = Trainer(**vars(trainer_args), logger=logger)
    trainer.fit(model=wrapper_model,
                datamodule=data_module)
