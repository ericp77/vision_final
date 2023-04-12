from unittest import TestCase
from main import transform
import torch


class Test(TestCase):
    def test_transform(self):
        img = torch.rand((64, 3, 256, 280))

        img = transform(img)
        self.assertEqual(img.size(), (3, 64, 224, 224))