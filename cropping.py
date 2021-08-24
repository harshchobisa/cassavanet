#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from PIL import Image

width = 800
height = 600
def getCropLines(j):
    if j == 0:
        left = 0
        top = 0
        right = 224
        bottom = 224

    if j == 1:
        left = width/2 - 112
        top = 0
        right = width/2 + 112
        bottom = 224

    if j == 2:
        left = width-224
        top = 0
        right = width
        bottom = 224


    if j == 3:
        left = 0
        top = height/2 - 112
        right = 224
        bottom = height/2 + 112

    if j == 4:
        left = width/2 - 112
        top = height/2 - 112
        right = width/2 + 112
        bottom = height/2 + 112

    if j == 5:
        left = width-224
        top = height/2 - 112
        right = width
        bottom = height/2 + 112

    if j == 6:
        left = 0
        top = height - 224
        right = 224
        bottom = height

    
    if j == 7:
        left = width/2 - 112
        top = height - 224
        right = width/2 + 112
        bottom = height

    if j == 8:
        left = width - 224
        top = height - 224
        right = width
        bottom = height
    
    return(left, top, right, bottom)



data = pd.read_csv("train.csv")

paths = list(data['image_id'])
labels = list(data['label'])

new_paths = []
new_labels = []

for i in range(len(paths)):
    print("working on image " + str(i), end='\r')
    path = paths[i]
    label = labels[i]
    
    new_labels += [label, label, label, label, label, label, label, label, label]
    im  = Image.open("./train_images/" + path)

    for j in range(9):
        left, top, right, bottom = getCropLines(j)
        cropped_image = im.crop((left, top, right, bottom))
        new_path_name = path[:-4] + "_" + str(j) + ".jpg"
        cropped_image.save("./train_images_cropped/" + new_path_name)
        new_paths.append(new_path_name)
    
    

new_data = pd.DataFrame()
new_data['image_id'] = new_paths
new_data['label'] = new_labels

new_data.to_csv("train_cropped.csv")
print("finished")




