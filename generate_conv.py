from model import Resnet50FPN
from PIL import Image
import os
import torch
import argparse
import math
import numpy as np
from utils import Transform0
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Counting code")
parser.add_argument("-dp", "--data_path", type=str, default='./chicken/', help="Path to the fish dataset")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU id")

args = parser.parse_args()
data_path = args.data_path
im_dir = data_path + 'single_chick/'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print(torch.cuda.is_available())
resnet50_conv = Resnet50FPN()
resnet50_conv.cuda()
resnet50_conv.eval()
feat_map_keys = ['map3', 'map4']
scale = 100 * (600.0 / 1280)
std_width = scale
std_height = scale
change_scales = [0.8, 0.9, 1.1, 1.2]

"""
generate features convolution kernels
"""


def get_conv():
    for i in range(1, 21):
        image_file = im_dir + str(i) + ".jpg"
        img = Image.open(image_file)
        img.load()

        sample = {'image': img}
        sample = Transform0(sample)
        img = sample['image'].cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        with torch.no_grad():
            Image_features = resnet50_conv(img.unsqueeze(0))

        for keys in feat_map_keys:
            image_feature = Image_features[keys]
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling = 16.0
            else:
                Scaling = 32.0
            to_w = int(std_width / Scaling)
            to_h = int(std_height / Scaling)
            if i == 1:
                if keys == 'map3':
                    map3_features = F.interpolate(image_feature, size=(to_h, to_w), mode='bilinear')
                else:
                    map4_features = F.interpolate(image_feature, size=(to_h, to_w), mode='bilinear')
            else:
                if keys == 'map3':
                    image_feature = F.interpolate(image_feature, size=(to_h, to_w), mode='bilinear')
                    map3_features = torch.cat((map3_features, image_feature), dim=0)
                else:
                    image_feature = F.interpolate(image_feature, size=(to_h, to_w), mode='bilinear')
                    map4_features = torch.cat((map4_features, image_feature), dim=0)
    return map3_features, map4_features


"""
Multi- scaling
"""


def change_scale(map3_conv, map4_conv):
    change_scales_conv = list()
    map3_h, map3_w = map3_conv.shape[2], map3_conv.shape[3]
    map4_h, map4_w = map4_conv.shape[2], map4_conv.shape[3]
    for scale in change_scales:
        if scale < 1:
            h_3 = math.floor(map3_h * scale)
            w_3 = math.floor(map3_w * scale)
            h_4 = math.floor(map4_h * scale)
            w_4 = math.floor(map4_w * scale)
        else:
            h_3 = math.ceil(map3_h * scale)
            w_3 = math.ceil(map3_w * scale)
            h_4 = math.ceil(map4_h * scale)
            w_4 = math.ceil(map4_w * scale)
        change_scales_conv.append(F.interpolate(map3_conv, size=(h_3, w_3), mode='bilinear'))
        change_scales_conv.append(F.interpolate(map4_conv, size=(h_4, w_4), mode='bilinear'))
    return change_scales_conv


map3_conv, map4_conv = get_conv()
change_scales_conv = list()
change_scales_conv = change_scale(map3_conv, map4_conv)

np.save("chicken/conv/map3_conv.npy", np.array(map3_conv.cpu()))
np.save("chicken/conv/map4_conv.npy", np.array(map4_conv.cpu()))
np.save("chicken/conv/map3_conv_08.npy", np.array(change_scales_conv[0].cpu()))
np.save("chicken/conv/map4_conv_08.npy", np.array(change_scales_conv[1].cpu()))
np.save("chicken/conv/map3_conv_09.npy", np.array(change_scales_conv[2].cpu()))
np.save("chicken/conv/map4_conv_09.npy", np.array(change_scales_conv[3].cpu()))
np.save("chicken/conv/map3_conv_11.npy", np.array(change_scales_conv[4].cpu()))
np.save("chicken/conv/map4_conv_11.npy", np.array(change_scales_conv[5].cpu()))
np.save("chicken/conv/map3_conv_12.npy", np.array(change_scales_conv[6].cpu()))
np.save("chicken/conv/map4_conv_12.npy", np.array(change_scales_conv[7].cpu()))
