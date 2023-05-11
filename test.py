from model import DensityRegressor, Resnet50FPN
from utils import MAPS, Transform, extract_features, visualize
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description="Counting code")
parser.add_argument("-dp", "--data_path", type=str, default='./chicken/', help="Path to the dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["test", "val"], help="what data split to evaluate on")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id")
parser.add_argument("-o", "--output-dir", type=str, default="./showimg/", help="Path to output")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'chicken.json'
data_split_file = data_path + 'Train_Test_Val_chicken.json'
im_dir = data_path + 'images'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
resnet50_conv = torch.load("./li4/backbone.pth")
resnet50_conv.cuda()
resnet50_conv.eval()

regressor = DensityRegressor(10, pool='mean')
regressor.load_state_dict(torch.load("./li4/generate_density.pth"))
regressor.cuda()
regressor.eval()

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)

cnt = 0
SAE = 0  # sum of absolute errors
SSE = 0  # sum of square errors

print("Evaluation on {} data".format(args.test_split))
im_ids = data_split[args.test_split]
pbar = tqdm(im_ids)
t_all = []
for im_id in pbar:
    t1 = time.time()
    anno = annotations[im_id]
    dots = np.array(anno['points'])

    image = Image.open('{}/{}'.format(im_dir, im_id))
    image.load()
    sample = {'image': image}
    sample = Transform(sample)
    image = sample['image']
    image = image.cuda()

    with torch.no_grad():
        output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), MAPS))

    gt_cnt = dots.shape[0]
    pred_cnt = output.sum().item()
    t2 = time.time()
    t_all.append(t2 - t1)
    cnt = cnt + 1
    err = abs(gt_cnt - pred_cnt)
    SAE += err
    SSE += err**2

    pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.\
                         format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE/cnt, (SSE/cnt)**0.5))
    print("")
    rslt_file = "{}/{}_myout.png".format(args.output_dir, im_id)
    """
    Save the visualization
    """
    visualize(image.detach().cpu(), output.detach().cpu(), rslt_file, dots)
print('average time:', np.mean(t_all) / 1)
print('average fps:', 1 / np.mean(t_all))
print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE/cnt, (SSE/cnt)**0.5))
