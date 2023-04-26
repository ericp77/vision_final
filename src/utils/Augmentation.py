import cv2
import numpy as np
import torch

from src.utils.utils import transform


class Augmentation:
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

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = transform(img)
        img = self.augment_data(img, self.sharpen, self.gamma)

        return img



