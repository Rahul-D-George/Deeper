from PIL import Image
import pandas as pd
import os
import numpy as np


def character_ages(type):
    character_df = pd.read_excel(r"Datasets/MoreCharacterInfo.xlsx")
    if type == 0:
        return list(map(lambda x: 1 if (x > 18) else 0, character_df["Age"].values.tolist()))
    if type == 1:
        ages = character_df["Age"].values.tolist()
        num_categories = 33
        ohl = np.zeros((len(ages), num_categories))
        for i, age in enumerate(ages):
            j = age
            ohl[i, j] = 1
        return ohl.T


def training_set_create(type = None): # Bin-class, Categorical, etc.
    base_names = os.listdir("img_set")
    dir_image_names = list(map(lambda x: os.path.join(os.getcwd(), "img_set\\" + x), base_names))
    image_data = []
    for image in dir_image_names:
        img_array = np.array(Image.open(image))
        rgb_array = img_array[:, :, :3]
        image_data.append(np.array(rgb_array))
    image_data = np.array(image_data)
    image_data = image_data.reshape(len(base_names), 185 * 185 * 3).T
    image_data = image_data / 255
    if type is None:
        return np.array(image_data), np.array(character_ages(0)).reshape((1, -1))
    elif type == "age":
        return np.array(image_data), character_ages(1)
    else:
        print("Invalid parameter passed.")
