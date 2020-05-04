import numpy as np

from PIL import Image

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torchvision.utils import save_image

to_tensor = transforms.ToTensor()

img_paths = ['./kfold/art_painting/dog/pic_001.jpg',
             './kfold/cartoon/dog/pic_001.jpg',
             './kfold/photo/dog/056_0002.jpg',
             './kfold/sketch/dog/5281.png',
             './kfold/art_painting/elephant/pic_001.jpg',
             './kfold/cartoon/elephant/pic_001.jpg',
             './kfold/photo/elephant/064_0001.jpg',
             './kfold/sketch/elephant/5921.png',
             './kfold/art_painting/giraffe/pic_001.jpg',
             './kfold/cartoon/giraffe/pic_001.jpg',
             './kfold/photo/giraffe/084_0001.jpg',
             './kfold/sketch/giraffe/7361.png',
             './kfold/art_painting/guitar/pic_001.jpg',
             './kfold/cartoon/guitar/pic_001.jpg',
             './kfold/photo/guitar/063_0001.jpg',
             './kfold/sketch/guitar/7601.png',
             ]

img_list = []
for i, img_path in enumerate(img_paths):
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

    img_list.append(to_tensor(img))



save_image(img_list,
                       'pacs_example.png', nrow=4)