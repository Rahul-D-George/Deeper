from PIL import Image
import pandas as pd
import os


def character_ages():
    character_df = pd.read_excel("MoreCharacterInfo.xlsx")
    ages = character_df["Age"].values.tolist()
    return ages


def training_set_create(args=None):
    if args is None:
        base_names = os.listdir("img_set")
        dir_image_names = list(map(lambda x: os.path.join(os.getcwd(), "img_set\\" + x), base_names))
        image_data = []
        for image in dir_image_names:
            image_data.append(Image.open(image))
        return [image_data, character_ages()]
    else:
        # Currently does nothing
        return None
