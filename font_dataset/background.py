import os
import random
from PIL import Image
import glob


__all__ = ["background_image_generator"]


def background_image_generator(path="./dataset/backgrounds"):
    # find all iamges (ends with jpg) in all folders using glob
    image_list = glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True)

    while True:
        yield random.choice(image_list)
