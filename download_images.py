import urllib.parse
import urllib.request
import os

import pandas as pd

from PIL import Image
import pandas as pd
import torch
from torchvision import transforms


HEADERS = [
    (
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    )
]


def main():
    df = pd.read_csv("./data/train_imputed.csv")

        # Load ResNet
    # resnet_full = torch.hub.load(
    #     "pytorch/vision:v0.6.0", "resnet18", pretrained=True
    # )
    # resnet = torch.nn.Sequential(*list(resnet_full.children())[:-1])
    # for param in resnet.parameters():
    #     param.requires_grad = False
    # resnet.eval()

    resnet = None

    for i, row in df.iterrows():
        print(i)
        path = './data/train_images_imputed/' + row['image_id']
        get_image_vector(path, row['image_id'], resnet)


def get_image_vector(image_path, image_id, resnet):
    processed_image_path = f"data/processed_images_imputed/{image_id}.pt"

    # if os.path.isfile(processed_image_path):
    #     input_tensor = torch.load(processed_image_path)


    #     resnet_output = resnet.forward(input_tensor.reshape([1, 3, 224, 224]))     
    #     resnet_output = resnet_output.reshape([512])
    #     # output = resnet.forward(input_tensor)
    #     processed_image_path_tensors = f"./data/processed_image_tensors_imputed/{image_id}.pt"
    #     torch.save(resnet_output, processed_image_path_tensors)

    #     return input_tensor

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    image = image.convert("RGB")

    input_tensor = preprocess(image)

    # resnet_output = resnet.forward(input_tensor.reshape([1, 3, 224, 224]))     
    # resnet_output = resnet_output.reshape([512])

    # processed_image_path_tensors = f"./data/processed_image_tensors_imputed/{image_id}.pt"

    # torch.save(resnet_output, processed_image_path_tensors)

    torch.save(input_tensor, processed_image_path)

main()
