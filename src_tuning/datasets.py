import os
import urllib.parse

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


class DailyBruinDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.list_data = []
        for i, row in self.data.iterrows():
            print(f"Row {i}", end="\r")
            row_data = get_image_vector(row['image_id'])
            self.list_data.append((row_data, row['label']))

    def __getitem__(self, index):
        image, label = self.list_data[index]
        return (image, label)

    def __len__(self):
        return len(self.list_data)


def get_image_vector(image_id):

    processed_image_path = f"./data/processed_image/{image_id}.pt"
    input_tensor = torch.load(processed_image_path)

    return input_tensor

