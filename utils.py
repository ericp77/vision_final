import torch
import torchvision.transforms as T


def transform(img: torch.Tensor):
    # resize 256x256
    resize = T.Resize((256, 256))
    img = resize(img)

    # random crop 224x224
    crop = T.RandomCrop((224, 224))
    img = crop(img)

    # normalize
    normalize = T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
    img = normalize(img.float())

    # permute
    img = torch.permute(img, (1, 0, 2, 3))
    return img
