#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from PIL import Image

data = pd.read_csv("data/train.csv")

paths = list(data['image_id'])
labels = list(data['label'])

new_paths = []
new_labels = []

names = ['rotated', 'vertical_flip', 'horizontal_flip']
functions = [Image.ROTATE_180, Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT]

for i in range(len(paths)):
    print("working on image " + str(i), end='\r')
    path = paths[i]
    label = labels[i]
    
    new_labels += [label, label, label, label]
    im  = Image.open("./data/train_images/" + path)
#     im  = Image.open("test.png")
    
    new_path_name = path[:-4] + "original.jpg"
    im.save("./data/train_images_imputed/" + new_path_name)
    new_paths.append(new_path_name)

    for j in range(3):
        new_path_name = path[:-4] + names[j] + ".jpg"
        new_image = im.transpose(functions[j])
        new_image.save("./data/train_images_imputed/" + new_path_name)
        
        new_paths.append(new_path_name)



new_data = pd.DataFrame()
new_data['image_id'] = new_paths
new_data['label'] = new_labels

new_data.to_csv("data/train_imputed.csv")
print("finished")




