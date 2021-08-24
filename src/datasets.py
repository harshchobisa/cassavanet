import os
import urllib.parse

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


class DailyBruinDataset(torch.utils.data.Dataset):
    def __init__(self, data, network_number):
        self.data = data
        self.list_data = [None for i in range(len(data))]
        self.image_source = ""

        if network_number == 0:
            self.image_source = "./data/processed_image_tensors_imputed/"
        else:
            self.image_source = "./data/processed_images/"


        # for i, row in self.data.iterrows():
        #     print(f"Row {i}", end="\r")
        #     row_data = self.get_image_vector(row['image_id'])
        #     self.list_data.append((row_data, row['label']))

    def __getitem__(self, index):

        if self.list_data[index] == None:
            row = self.data.iloc[index]
            row_data = self.get_image_vector(row['image_id'])
            self.list_data[index] = (row_data, row['label'])
        
        image, label = self.list_data[index]

        return (image, label)

    def __len__(self):
        return len(self.list_data)


    def get_image_vector(self, image_id):

        processed_image_path = f"{self.image_source}{image_id}.pt"
        input_tensor = torch.load(processed_image_path)

        return input_tensor

