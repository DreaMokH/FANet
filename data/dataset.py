import os
import time
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils.equi_to_cube import Equi2Cube


class SOD360Dataset(Dataset):
    def __init__(self, img_dir, list_path, input_size, train=True):
        self.img_dir = img_dir
        self.list_path = list_path
        self.input_size = input_size
        self.data = []
        file_list = [line.rstrip().split(' ') for line in tuple(open(self.list_path, "r"))]

        for file_id in file_list:
            image_id = file_id[0].split("/")[-1].split(".")[0]
            img_file = os.path.join(self.img_dir, file_id[0][1:])
            label_file = os.path.join(self.img_dir, file_id[1][1:])

            self.data.append({"img": img_file, "label": label_file,"name": image_id})

        self.train = train

        self.normalization = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
        self.label2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        imgfile = self.data[index]
        name = imgfile["name"]
        equi_img = Image.open(imgfile["img"]).convert('RGB')
        label = Image.open(imgfile["label"]).convert('L')
        img_size = [equi_img.width, equi_img.height]

        equi_img = equi_img.resize(self.input_size, resample=Image.LANCZOS)
        equi_label = label.resize(self.input_size, resample=Image.LANCZOS)

        label = np.array(equi_label)
        label = self.label2tensor(label)

        label_list = [label]

        equi_array = np.array(equi_img) / 255.0
        cube_img = Equi2Cube(self.input_size[1] // 2, equi_array)
        output_cubedic = cube_img.to_cube(equi_array)
        cube_B = Image.fromarray(np.uint8(output_cubedic[0]* 255.0))
        cube_D = Image.fromarray(np.uint8(output_cubedic[1]* 255.0))
        cube_F = Image.fromarray(np.uint8(output_cubedic[2]* 255.0))
        cube_L = Image.fromarray(np.uint8(output_cubedic[3]* 255.0))
        cube_R = Image.fromarray(np.uint8(output_cubedic[4]* 255.0))
        cube_T = Image.fromarray(np.uint8(output_cubedic[5]* 255.0))

        equi_img = self.normalization(equi_img)
        cube_B = self.normalization(cube_B)
        cube_D = self.normalization(cube_D)
        cube_F = self.normalization(cube_F)
        cube_L = self.normalization(cube_L)
        cube_R = self.normalization(cube_R)
        cube_T = self.normalization(cube_T)
        input_list = [equi_img, cube_B, cube_D, cube_F, cube_L, cube_R, cube_T]

        return input_list, label_list, name, img_size

