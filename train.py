import torch.nn as nn
from model import Resnet50FPN, DensityRegressor, weights_normal_init
from utils import MAPS, Transform, TransformTrain, extract_features, ExtraLoss
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists, join
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description="Counting code")
parser.add_argument("-dp", "--data_path", type=str, default='./chicken/', help="Path to the dataset")
parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Path to save models")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"],
                    help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU id")
parser.add_argument("-wm", "--extra_loss_weight", type=float, default=1e-9, help="weight for extra Loss")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5, help="learning rate")
args = parser.parse_args()

data_path = args.data_path
anno_file = data_path + 'chicken_annotation.json'
data_split_file = data_path + 'Train_Test_Val_chicken.json'
im_dir = data_path + 'images'
gt_dir = data_path + 'annocation'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

resnet50_conv = Resnet50FPN()
resnet50_conv.train()
resnet50_conv.cuda()
optimizer0 = optim.Adam(resnet50_conv.parameters(), lr=args.learning_rate)
scheduler0 = lr_scheduler.StepLR(optimizer0, step_size=20, gamma=0.9, verbose=True)

regressor = DensityRegressor(10, pool='mean')
weights_normal_init(regressor, dev=0.001)
regressor.train()
regressor.cuda()
optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9, verbose=True)

with open(anno_file) as f:
    annotations = json.load(f)

with open(data_split_file) as f:
    data_split = json.load(f)


def train():
    optimizer0.zero_grad()
    print("Training on chicken train set data")
    im_ids = data_split['train']
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    pbar = tqdm(im_ids)
    cnt = 0
    for im_id in pbar:
        cnt += 1
        anno = annotations[im_id]
        boxes = anno['boxs']

        rects = list()
        for bbox in boxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[1][0]
            y2 = bbox[1][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()

        density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')
        sample = {'image': image, 'boxes': rects, 'gt_density': density}
        sample = TransformTrain(sample)
        image, boxes, gt_density = sample['image'].cuda(), sample['boxes'].cuda(), sample['gt_density'].cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        features = extract_features(resnet50_conv, image.unsqueeze(0), MAPS)

        optimizer.zero_grad()

        output = regressor(features)

        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2], output.shape[3]), mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0:
                gt_density = gt_density * (orig_count / new_count)

        loss = criterion(output, gt_density)
        extraloss = args.extra_loss_weight * ExtraLoss(output, gt_density, boxes)
        loss = loss + extraloss
        loss.backward()
        optimizer0.step()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        pbar.set_description('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f} Best VAL MAE: {:5.2f}, RMSE: {:5.2f}'.format(
                gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae / cnt, (train_rmse / cnt) ** 0.5, best_mae, best_rmse))
        print("")
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids)) ** 0.5
    return train_loss, train_mae, train_rmse


def eval():
    cnt = 0
    SAE = 0
    SSE = 0

    print("Evaluation on {} data".format(args.test_split))
    im_ids = data_split[args.test_split]
    pbar = tqdm(im_ids)
    for im_id in pbar:
        anno = annotations[im_id]
        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(im_dir, im_id))
        image.load()
        sample = {'image': image}
        sample = Transform(sample)
        image = sample['image'].cuda()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        with torch.no_grad():
            output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), MAPS))

        gt_cnt = dots.shape[0]
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err ** 2

        pbar.set_description(
            '{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'. \
            format(im_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE / cnt, (SSE / cnt) ** 0.5))
        print("")

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(args.test_split, SAE / cnt, (SSE / cnt) ** 0.5))
    return SAE / cnt, (SSE / cnt) ** 0.5


best_mae, best_rmse = 1e7, 1e7
stats = list()
for epoch in range(0, args.epochs):
    resnet50_conv.train()
    regressor.train()
    train_loss, train_mae, train_rmse = train()
    scheduler0.step()
    scheduler.step()
    resnet50_conv.eval()
    regressor.eval()
    val_mae, val_rmse = eval()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(args.output_dir, "stats" + ".txt")
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write("%s\n" % ','.join([str(x) for x in s]))
    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = args.output_dir + '/' + "generate_density.pth"
        model_name0 = args.output_dir + '/' + "backbone.pth"
        torch.save(resnet50_conv, model_name0)
        torch.save(regressor.state_dict(), model_name)

    print("Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} ".format(
            epoch + 1, stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))
