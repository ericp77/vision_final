import torch

from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader
from model import S3D
from utils import transform


if __name__ == "__main__":
    train_dataset = HMDB51(
        root="dataset/hmdb51",
        annotation_path="dataset/splits",
        frames_per_clip=64,
        train=True,
        transform=transform,
        fold=1,
        output_format="TCHW"
    )

    # validation_dataset = HMDB51(
    # root="dataset/hmdb51",
    # annotation_path="dataset/splits",
    # frames_per_clip=64,
    # train=False,
    # transform =transform,
    # fold=1,
    # output_format="TCHW"
    # )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True
    )

    # from: https://github.com/kylemin/S3D
    model = S3D(num_class=len(train_dataset.classes))

    for i, (video, audio, label) in enumerate(train_dataloader):
        print(video.shape)
        print(label)
        break
