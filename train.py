from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2, _ in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100



def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def linear_predict(feature, linear_model):
    pred_scores = linear_model(feature)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

def train_linear(linear,net,data_loader, train_optimizer, epoch, args):
    linear.train()
    net.eval()
    adjust_learning_rate(train_optimizer, epoch, args)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_num, total_correct = 0.0, 0, 0
    train_bar = tqdm(data_loader)
    for im_1, im_2,target in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        #print(im_1.shape)
        feature = net(im_1)
   
        predict = linear(feature)
 
        loss = criterion(predict, target)
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        _, predicted_labels = torch.max(predict, 1)
        total_correct += (predicted_labels == target).sum().item()
        
        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num, total_correct / total_num * 100))


    return total_loss / total_num, total_correct / total_num * 100
    
    
def test_linear(linear, net, test_loader, epoch, args):
    linear.eval()
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_num, total_correct = 0.0, 0, 0
    test_bar = tqdm(test_loader)
    for im,target in test_bar:
        im = im.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        feature = net(im)
   
        predict = linear(feature)
        loss = criterion(predict, target)
        
        _, predicted_labels = torch.max(predict, 1)
        total_correct += (predicted_labels == target).sum().item()
        total_num += test_loader.batch_size
        total_loss += loss.item() * test_loader.batch_size
        test_bar.set_description('Test Epoch: [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, args.epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100