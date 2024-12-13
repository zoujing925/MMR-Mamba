from __future__ import print_function, division
import numpy as np
import pandas as pd
import os, glob
import random
from skimage import transform
from PIL import ImageChops, Image


def central_crop(image, border):
    width, height = image.size

    left = border
    top = border
    right = width - border
    bottom = height - border

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image
    

directory = '/home/sh2/users/zj/MRI/BRATS_100patients/image_100patients_8X/'
out_directory = '/home/sh2/users/zj/MRI/BRATS_100patients/image_100patients_8X_224/'
if not os.path.exists(out_directory):
    os.makedirs(out_directory)
    
t1_images = glob.glob(os.path.join(directory, '*t1.png'))
# print("t1_images:{}".format(t1_images))
for filename in t1_images:
    # print("filename:{}".format(filename))
    image = Image.open(filename)
    image = central_crop(image, 8)
    savepath = os.path.join(out_directory, filename.split('/')[-1])
    image.save(savepath)
    # print("filename:{}".format(savepath))
    # print("image.size:{}".format(image.size))

    if image.size[0] != 224 or image.size[1] != 224:
        print("filename:{}".format(filename))


# t2_images = glob.glob(os.path.join(directory, '*t2.png'))
# for filename in t2_images:
#     image = Image.open(filename)
#     image = central_crop(image, 8)
#     savepath = os.path.join(out_directory, filename.split('/')[-1])
#     image.save(savepath)
#     # print("filename:{}".format(savepath))
#     # print("image.size:{}".format(image.size))


#     if image.size[0] != 224 or image.size[1] != 224:
#         print("filename:{}".format(filename))

# t2_undermri = glob.glob(os.path.join(directory, '*t2_8X_undermri.png'))
# for filename in t2_undermri:
#     image = Image.open(filename)
#     image = central_crop(image, 8)
#     savepath = os.path.join(out_directory, filename.split('/')[-1])
#     image.save(savepath)
#     # print("filename:{}".format(savepath))
#     if image.size[0] != 224 or image.size[1] != 224:
#         print("filename:{}".format(filename))
#     # print("image.size:{}".format(image.size))