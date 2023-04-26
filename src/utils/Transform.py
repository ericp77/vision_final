import cv2
import numpy as np
import torch

import torchvision.transforms as T


class Transform:
    def __init__(self, sharpen: bool = False, gamma: bool = False):
        self.sharpen = sharpen
        self.gamma = gamma

    def augment_data(self, img, sharpen=True, gamma_adjust=True, gamma_value=1.5):
        def sharpen_image(image):
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened

        def gamma_correction(image, gamma=1.0):
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        numpy_array = img.numpy()
        augmented_frames = []

        for frame_idx in range(img.shape[0]):
            frame = numpy_array[frame_idx]

            if sharpen:
                frame = sharpen_image(frame)

            if gamma_adjust:
                frame = gamma_correction(frame, gamma=gamma_value)

            augmented_frames.append(frame)

        return torch.tensor(augmented_frames)

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        normalize = T.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5])
        return normalize(img.float())

    def crop(self, img: torch.Tensor):
        # resize 256x256
        resize = T.Resize((256, 256))
        img = resize(img)

        # random crop 224x224
        crop = T.RandomCrop((224, 224))
        img = crop(img)
        return img

    def permute(self, img: torch.Tensor):
        return torch.permute(img, (1, 0, 2, 3))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = self.crop(img)
        if self.sharpen or self.gamma:
            img = self.augment_data(img, self.sharpen, self.gamma)
        img = self.normalize(img)
        img = self.permute(img)

        return img



