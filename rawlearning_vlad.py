from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
import netvlad

from utils import create_logger

import os
import time

dtype = torch.cuda.FloatTensor  # run on GPU

import multiprocessing
multiprocessing.set_start_method('spawn', True)

from scipy.io import savemat,loadmat

import json
from img_gist_feature.utils_gist import *
from img_gist_feature.utils_tool import *
import matplotlib.pyplot as plt
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use gpu

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batchSize', type=int, default=12, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=96, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=3000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='/home/annora/NetVLAD/output/data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='/home/annora/NetVLAD/output/runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='/home/annora/NetVLAD/output/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='pittsburgh', 
        help='Dataset to use', choices=['pittsburgh', 'robotcar'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--fromtask', type=str, default='', help='from which resume task run')
parser.add_argument('--premodel', type=str, default='imagenet', choices=['imagenet', 'places'], help='which pretrained models will be used')
parser.add_argument('--contilearn', action='store_true', help='Use continual learning')



def train(epoch):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        # print('====> Building Cache')
        logger.info('====> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache0.hdf5')
        logger.info("feature cache--->>> {}".format(train_set.cache))
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", 
                    [len(whole_train_set), pool_size], 
                    dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    image_encoding, _ = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding) 
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=cuda)

        # print('Allocated:', torch.cuda.memory_allocated())
        logger.info('Allocated:{} {}'.format(torch.cuda.memory_allocated()/(1024*1024), 'MB'))
        # print('Cached:', torch.cuda.memory_cached())
        logger.info('Cached:{} {}'.format(torch.cuda.memory_cached()/(1024*1024), 'MB'))

        model.train()
        for iteration, (query, positives, negatives, 
                negCounts, indices) in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding, x_list = model.encoder(input, B)
            vlad_encoding = model.pool(image_encoding) 

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()

            lamda = iteration / len(train_set) / opt.nEpochs + (epoch - 1)/opt.nEpochs

            alpha_array = [1.0 * 0.00001 ** lamda]
            # alpha_array = [1.0]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    with torch.no_grad():
                        for i in range(Ho):
                            for j in range(Wo):
                                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                                r = x[:, :, i * S: i * S + HH, j *
                                    S: j * S + WW].contiguous().view(1, -1)
                                # r = r[:, range(r.shape[1] - 1, -1, -1)]
                                k = torch.mm(p, torch.t(r))
                                p.sub_(torch.mm(k, torch.t(k)) /
                                       (alpha + torch.mm(r, k)))
                                # Returns the current GPU memory usage by tensors in bytes for a given device.
                                # print('Allocated:', torch.cuda.memory_allocated()/1073741824, 'GB')
                                memory_allocated = torch.cuda.memory_allocated()/1073741824
                    w.grad.data = torch.mm(w.grad.data.view(
                        F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))

            if opt.contilearn and opt.arch == 'alexnet':
                for n, w in model.named_parameters():
                    if n == 'encoder.conv1.0.weight':
                        # logger.info('{} grad_predata\n {}'.format(n, w.grad.data))
                        pro_weight(pc_dict['Pc1'], x_list[0], w,
                                   alpha=alpha_array[0], stride=2)
                        # logger.info('{} grad_behdata\n {}'.format(n, w.grad.data))
                    if n == 'encoder.conv2.0.weight':
                        # logger.info('{} grad_predata\n {}'.format(n, w.grad.data))
                        pro_weight(pc_dict['Pc2'], x_list[1], w,
                                   alpha=alpha_array[0], stride=2)
                        # logger.info('{} grad_behdata\n {}'.format(n, w.grad.data))
                    if n == 'encoder.conv3.0.weight':
                        # logger.info('{} grad_predata\n {}'.format(n, w.grad.data))
                        pro_weight(pc_dict['Pc3'], x_list[2], w,
                                   alpha=alpha_array[0], stride=2)
                        # logger.info('{} grad_behdata\n {}'.format(n, w.grad.data))
                    if n == 'encoder.conv4.0.weight':
                        # logger.info('{} grad_predata\n {}'.format(n, w.grad.data))
                        pro_weight(pc_dict['Pc4'], x_list[3], w,
                                   alpha=alpha_array[0], stride=2)
                        # logger.info('{} grad_behdata\n {}'.format(n, w.grad.data))
                    if n == 'encoder.conv5.0.weight':
                        # logger.info('{} grad_predata\n {}'.format(n, w.grad.data))
                        pro_weight(pc_dict['Pc5'], x_list[4], w,
                                   alpha=alpha_array[0], stride=2)
                        # logger.info('{} grad_behdata\n {}'.format(n, w.grad.data))
            elif opt.contilearn and opt.arch == 'vgg16':
                for n, w in model.named_parameters():
                    if n == 'encoder.conv1.0.weight':
                        pro_weight(pc_dict['PC1'], x_list[0], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv2.0.weight':
                        pro_weight(pc_dict['Pc2'], x_list[1], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv3.0.weight':
                        pro_weight(pc_dict['Pc3'], x_list[2], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv4.0.weight':
                        pro_weight(pc_dict['Pc4'], x_list[3], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv5.0.weight':
                        pro_weight(pc_dict['Pc5'], x_list[4], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv6.0.weight':
                        pro_weight(pc_dict['Pc6'], x_list[5], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv7.0.weight':
                        pro_weight(pc_dict['Pc7'], x_list[6], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv8.0.weight':
                        pro_weight(pc_dict['Pc8'], x_list[7], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv9.0.weight':
                        pro_weight(pc_dict['Pc9'], x_list[8], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv10.0.weight':
                        pro_weight(pc_dict['Pc10'], x_list[9], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv11.0.weight':
                        pro_weight(pc_dict['Pc11'], x_list[10], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv12.0.weight':
                        pro_weight(pc_dict['Pc12'], x_list[11], w,
                                   alpha=alpha_array[0], stride=2)
                    if n == 'encoder.conv13.0.weight':
                        pro_weight(pc_dict['Pc13'], x_list[12], w,
                                   alpha=alpha_array[0], stride=2)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                # print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                #     nBatches, batch_loss), flush=True)
                logger.info("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                # print('Allocated:', torch.cuda.memory_allocated())
                logger.info('Allocated:{} {}'.format(torch.cuda.memory_allocated()/(1024*1024), 'MB'))
                # print('Cached:', torch.cuda.memory_cached())
                logger.info('Cached:{} {}'.format(torch.cuda.memory_cached()/(1024*1024), 'MB'))

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
    #         flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test1(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        # print('====> Extracting Features')
        logger.info('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding, _ = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                # print("==> Batch ({}/{})".format(iteration, 
                #     len(test_data_loader)), flush=True)
                logger.info("==> Batch ({}/{})".format(iteration, len(test_data_loader)))

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    # print('====> Building faiss index')
    logger.info('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    # print('====> Calculating recall @ N')
    logger.info('====> Calculating recall @ N')

    numq = eval_set.dbStruct.numQ
    # n_values = [1,2,3,4,5,10,15,20,25,numq // 50,numq // 40, numq // 30, numq // 20, numq // 10, numq // 5, numq // 4, numq // 3, numq // 2, numq]
    n_values = [1,2,3,4,5,10,15,20,25,3000]

    # _, predictions = faiss_index.search(qFeat, max(n_values)) 
    dd, predictions = faiss_index.search(qFeat, eval_set.dbStruct.numDb) 


    maxd = np.max(dd)
    mind = np.min(dd)

    thresh_values = mind + [0.0001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


    # for each query get those within threshold distance
    gt = eval_set.getPositives() 

    true_positive_at_n = np.zeros(len(thresh_values))
    false_positive_at_n = np.zeros(len(thresh_values))

    # calculate the false_positive
    for qIx, pred in enumerate(predictions):
        for i, thresh in enumerate(thresh_values):
            # count = 0
            # for ii, predd in enumerate(dd[qIx]):
            #     if predd > thresh:
            #         count = ii
            #         break
            bool_n = dd[qIx] <= thresh
            count = np.sum(bool_n)
            pred_set = set(pred[:count].tolist())
            gt_set = set(gt[qIx].tolist())
            interset = pred_set & gt_set
            false_positive_at_n[i] += count - len(interset)
            # for _, predd in enumerate(pred[:count]):
            #     if predd not in gt[qIx]:
            #         false_positive_at_n[i] += 1

    # calculate the true_positive where the query result 
    for qIx, pred in enumerate(predictions):
        for i, thresh in enumerate(thresh_values):
            bool_n = dd[qIx] <= thresh
            count = np.sum(bool_n)
            if np.any(np.in1d(pred[:count], gt[qIx])):
                true_positive_at_n[i:] += 1
                break

    recall_at_n = true_positive_at_n / eval_set.dbStruct.numQ
    precision_at_n = true_positive_at_n / (true_positive_at_n + false_positive_at_n)


    # correct_at_n = np.zeros(len(n_values))
    # false_positive_at_n = np.zeros(len(n_values))
    # #TODO can we do this on the matrix in one go?
    # for qIx, pred in enumerate(predictions):
    #     for i,n in enumerate(n_values):
    #         # if in top N then also in top NN, where NN > N
    #         if np.any(np.in1d(pred[:n], gt[qIx])):
    #             correct_at_n[i:] += 1
    #             break
    # for qIx, pred in enumerate(predictions):
    #     for i,n in enumerate(n_values):
    #         # if in top N then also in top NN, where NN > N
    #         for _, predd in enumerate(pred[:n]):
    #             if predd not in gt[qIx]:
    #                 false_positive_at_n[i] += 1
    # recall_at_n = correct_at_n / eval_set.dbStruct.numQ
    # precision_at_n = correct_at_n / (correct_at_n + false_positive_at_n)

    recalls = {} #make dict for output
    precisions = {}
    for i,n in enumerate(thresh_values):
        recalls[n] = recall_at_n[i]
        precisions[n] = precision_at_n[i]
        # print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        logger.info("====> Recall: {:.4f}, Precision: {:.4f}".format(recall_at_n[i], precision_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall_' + str(n), recall_at_n[i], epoch)

    savemat('./mat/NPD_CS5.mat', {'recalls':recall_at_n,'precisions':precision_at_n})

    return 
    
def testd(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=1, shuffle=False)

    gist_helper = GistUtils()

    pool_size = 512

    model.eval()
    with torch.no_grad():
        # print('====> Extracting Features')
        logger.info('====> Extracting Features')
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            # input = input.to(device)
            img = input.mul(255).byte()
            np_img = img.cpu().numpy().squeeze(0).transpose((1,2,0))
            np_gist = gist_helper.get_gist_vec(np_img, mode="gray")
            np_gist_L2Norm = np_l2norm(np_gist)
            dbFeat[indices.detach().numpy(), :] = np_gist_L2Norm
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                # print("==> Batch ({}/{})".format(iteration, 
                #     len(test_data_loader)), flush=True)
                logger.info("==> Batch ({}/{})".format(iteration, len(test_data_loader)))

            del input
    del test_data_loader

    savemat('./mat/Night_Feats.mat', {'dbFeat':dbFeat})
    dnmat = loadmat('./mat/Day_Feats.mat')
    dbFeat = dnmat['dbFeat']
    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    qFeat = np.ascontiguousarray(qFeat)
    dbFeat = np.ascontiguousarray(dbFeat)

    # print('====> Building faiss index')
    logger.info('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    # print('====> Calculating recall @ N')
    logger.info('====> Calculating recall @ N')

    n_values = [1,2,3,4,5,10,15,20,25]

    # _, predictions = faiss_index.search(qFeat, max(n_values)) 
    dd, predictions = faiss_index.search(qFeat, eval_set.dbStruct.numDb) 

    maxd = np.max(dd)
    mind = np.min(dd)

    thresh_values = mind + [0.0001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # for each query get those within threshold distance
    gt = eval_set.getPositives() 

    # true_positive_at_n = np.zeros(len(thresh_values))
    # false_positive_at_n = np.zeros(len(thresh_values))

    # # calculate the false_positive
    # for qIx, pred in enumerate(predictions):
    #     for i, thresh in enumerate(thresh_values):
    #         bool_n = dd[qIx] <= thresh
    #         count = np.sum(bool_n)
    #         pred_set = set(pred[:count].tolist())
    #         gt_set = set(gt[qIx].tolist())
    #         interset = pred_set & gt_set
    #         false_positive_at_n[i] += count - len(interset)

    # # calculate the true_positive where the query result 
    # for qIx, pred in enumerate(predictions):
    #     for i, thresh in enumerate(thresh_values):
    #         bool_n = dd[qIx] <= thresh
    #         count = np.sum(bool_n)
    #         if np.any(np.in1d(pred[:count], gt[qIx])):
    #             true_positive_at_n[i:] += 1
    #             break

    # recall_at_n = true_positive_at_n / eval_set.dbStruct.numQ
    # precision_at_n = true_positive_at_n / (true_positive_at_n + false_positive_at_n)

    # recalls = {} #make dict for output
    # precisions = {}
    # for i,n in enumerate(thresh_values):
    #     recalls[n] = recall_at_n[i]
    #     precisions[n] = precision_at_n[i]
    #     # print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
    #     logger.info("====> Recall: {:.4f}, Precision: {:.4f}".format(recall_at_n[i], precision_at_n[i]))
    #     if write_tboard: writer.add_scalar('Val/Recall_' + str(n), recall_at_n[i], epoch)

    # savemat('./mat/DN_D_Pre.mat', {'recalls':recall_at_n,'precisions':precision_at_n})

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        # print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        logger.info("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall_' + str(n), recall_at_n[i], epoch)
    
    savemat('./mat/Night_Recall_GIST.mat', {'recalls':recall_at_n})

    return recalls

def test(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=eval_set,
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        # print('====> Extracting Features')
        logger.info('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad':
            pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding, _ = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(),
                                  :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                # print("==> Batch ({}/{})".format(iteration,
                #     len(test_data_loader)), flush=True)
                logger.info("==> Batch ({}/{})".format(iteration,
                                                       len(test_data_loader)))

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    # print('====> Building faiss index')
    logger.info('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    # print('====> Calculating recall @ N')
    logger.info('====> Calculating recall @ N')
    n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]


    # _, predictions = faiss_index.search(qFeat, max(n_values))
    
    start_time = time.time()
    _, predictions = faiss_index.search(qFeat, 5)
    cost_time = time.time()-start_time
    print(cost_time)

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        # print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        logger.info("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall_' + str(n), recall_at_n[i], epoch)
    
    savemat('./mat/NPD_N_Beh_Recall_CS3.mat', {'recalls':recall_at_n})

    return recalls


def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            logger.info('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    # print("==> Batch ({}/{})".format(iteration, 
                    #     ceil(nIm/opt.cacheBatchSize)), flush=True)
                    logger.info("==> Batch ({}/{})".format(iteration, ceil(nIm/opt.cacheBatchSize)))
                del input, image_descriptors
        
        # print('====> Clustering..')
        logger.info('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        # print('====> Storing centroids', kmeans.centroids.shape)
        logger.info('====> Storing centroids {}'.format(kmeans.centroids.shape))
        h5.create_dataset('centroids', data=kmeans.centroids)
        # print('====> Done!')
        logger.info('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    try:
        opt = parser.parse_args()
        
        logger = create_logger('output', opt)

        restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
                'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
                'margin', 'seed', 'patience']
        if opt.resume:
            flag_file = join(opt.resume, 'checkpoints', 'flags.json')
            if exists(flag_file):
                with open(flag_file, 'r') as f:
                    stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                    to_del = []
                    for flag, val in stored_flags.items():
                        for act in parser._actions:
                            if act.dest == flag[2:]:
                                # store_true / store_false args don't accept arguments, filter these 
                                if type(act.const) == type(True):
                                    if val == str(act.default):
                                        to_del.append(flag)
                                    else:
                                        stored_flags[flag] = ''
                    for flag in to_del: del stored_flags[flag]

                    train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                    print('Restored flags:', train_flags)
                    opt = parser.parse_args(train_flags, namespace=opt)

        # print(opt)
        logger.info(opt)

        if opt.dataset.lower() == 'pittsburgh':
            import pittsburgh_continual as dataset
        elif opt.dataset.lower() =='robotcar':
            import OxfordRobotCar as dataset
        else:
            logger.exception('Unknown dataset')
            raise Exception('Unknown dataset')

        cuda = not opt.nocuda
        if cuda and not torch.cuda.is_available():
            logger.error("No GPU found, please run with --nocuda")
            raise Exception("No GPU found, please run with --nocuda")

        device = torch.device("cuda" if cuda else "cpu")

        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if cuda:
            torch.cuda.manual_seed(opt.seed)


        trainfiles= ['oxRobocar_single_train.mat', 'oxRobocar_doubleN_train.mat', 'pitts30k_train.mat']
        valfiles = ['oxRobocar_single_val.mat', 'oxRobocar_doubleN_val.mat', 'pitts30k_val.mat']
        testfiles = ['oxRobocar_single_test.mat', 'oxRobocar_doubleN_test.mat', 'pitts30k_test.mat', 'oxRobocar_doubleDN_test.mat', 'XY_doubleNP_test.mat', 'XY_tripleDPN_test.mat', 'XY_doubleDP_test.mat']

        trainfile = trainfiles[0]
        valfile = valfiles[0]
        # testfile = valfiles[0]
        testfile = testfiles[5]

        if (opt.mode == 'test') and ('XY' in testfile):
            import datasetAbsPath as dataset


        # print('===> Loading dataset(s)')
        logger.info('===> Loading dataset(s)')
        if opt.mode.lower() == 'train':
            logger.info("trainfile ------>>>>>> {}".format(trainfile))
            whole_train_set = dataset.get_whole_training_set(matfile=trainfile)
            whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                    num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                    pin_memory=cuda)

            train_set = dataset.get_training_query_set(opt.margin, matfile=trainfile)

            # print('====> Training query set:', len(train_set))
            logger.info('====> Training set: {}, query set count: {}'.format(trainfile, len(train_set)))
            whole_test_set = dataset.get_whole_val_set(matfile=valfile)
            # print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
            logger.info('===> Evaluating on val set: {}, query count: {}'.format(valfile, whole_test_set.dbStruct.numQ))
        elif opt.mode.lower() == 'test':
            logger.info("testfile ------>>>>>> {}".format(testfile))
            if opt.split.lower() == 'test':
                whole_test_set = dataset.get_whole_test_set(matfile=testfile)
                # print('===> Evaluating on test set')
                logger.info('===> Evaluating on test set')
            elif opt.split.lower() == 'test250k':
                whole_test_set = dataset.get_250k_test_set()
                # print('===> Evaluating on test250k set')
                logger.info('===> Evaluating on test250k set')
            elif opt.split.lower() == 'train':
                whole_test_set = dataset.get_whole_training_set()
                # print('===> Evaluating on train set')
                logger.info('===> Evaluating on train set')
            elif opt.split.lower() == 'val':
                whole_test_set = dataset.get_whole_val_set()
                # print('===> Evaluating on val set')
                logger.info('===> Evaluating on val set')
            else:
                logger.error('Unknown dataset split: {}'.format(opt.split))
                raise ValueError('Unknown dataset split: ' + opt.split)
            logger.info('====> Query count:{}'.format(whole_test_set.dbStruct.numQ))
            # print('====> Query count:', whole_test_set.dbStruct.numQ)
        elif opt.mode.lower() == 'cluster':
            whole_train_set = dataset.get_whole_training_set(onlyDB=True, matfile=trainfile)

        # print('===> Building model')
        logger.info('===> Building model')

        pretrained = not opt.fromscratch
        if opt.arch.lower() == 'alexnet':
            # encoder_dim = 256
            # encoder = models.alexnet(pretrained=pretrained)
            # # capture only features and remove last relu and maxpool
            # layers = list(encoder.features.children())[:-2]

            # if pretrained:
            #     # if using pretrained only train conv5
            #     for l in layers[:-1]:
            #         for p in l.parameters():
            #             p.requires_grad = False
            import alexnet
            encoder_dim = 256
            # encoder = models.alexnet(pretrained=pretrained)
            # 加载ImageNet预训练的模型参数
            encoder = alexnet.AlexNet()
            # print(encoder)
            logger.info('encoder moder:\n{}'.format(encoder))
            model_dict = encoder.state_dict()
            if pretrained and opt.premodel == 'imagenet':
                checkpoint_alex = torch.load(
                    '/home/annora/.torch/models/alexnet-owt-4df8aa71.pth', map_location='cpu')
                model_dict['conv1.0.weight'] = checkpoint_alex['features.0.weight']
                model_dict['conv1.0.bias'] = checkpoint_alex['features.0.bias']
                model_dict['conv2.0.weight'] = checkpoint_alex['features.3.weight']
                model_dict['conv2.0.bias'] = checkpoint_alex['features.3.bias']
                model_dict['conv3.0.weight'] = checkpoint_alex['features.6.weight']
                model_dict['conv3.0.bias'] = checkpoint_alex['features.6.bias']
                model_dict['conv4.0.weight'] = checkpoint_alex['features.8.weight']
                model_dict['conv4.0.bias'] = checkpoint_alex['features.8.bias']
                model_dict['conv5.0.weight'] = checkpoint_alex['features.10.weight']
                model_dict['conv5.0.bias'] = checkpoint_alex['features.10.bias']
            elif pretrained and opt.premodel == 'places':
                checkpoint_alex1 = torch.load(
                    '/home/annora/.torch/models/alexnet_places365.pth.tar', map_location='cpu')
                checkpoint_alex = checkpoint_alex1['state_dict']
                model_dict['conv1.0.weight'] = checkpoint_alex['features.module.0.weight']
                model_dict['conv1.0.bias'] = checkpoint_alex['features.module.0.bias']
                model_dict['conv2.0.weight'] = checkpoint_alex['features.module.3.weight']
                model_dict['conv2.0.bias'] = checkpoint_alex['features.module.3.bias']
                model_dict['conv3.0.weight'] = checkpoint_alex['features.module.6.weight']
                model_dict['conv3.0.bias'] = checkpoint_alex['features.module.6.bias']
                model_dict['conv4.0.weight'] = checkpoint_alex['features.module.8.weight']
                model_dict['conv4.0.bias'] = checkpoint_alex['features.module.8.bias']
                model_dict['conv5.0.weight'] = checkpoint_alex['features.module.10.weight']
                model_dict['conv5.0.bias'] = checkpoint_alex['features.module.10.bias']
            encoder.load_state_dict(model_dict)
        elif opt.arch.lower() == 'vgg16':
            # encoder_dim = 512
            # encoder = models.vgg16(pretrained=pretrained)
            # # capture only feature part and remove last relu and maxpool
            # layers = list(encoder.features.children())[:-2]

            # if pretrained:
            #     # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            #     for l in layers[:-5]: 
            #         for p in l.parameters():
            #             p.requires_grad = False
            import vgg16
            encoder_dim = 512
            encoder = vgg16.VGG16()
            logger.info('encoder moder:\n{}'.format(encoder))
            if pretrained:
                checkpoint_vgg16 = torch.load('/home/annora/.torch/models/vgg16-397923af.pth')
                model_dict = encoder.state_dict()
                model_dict['conv1.0.weight'] = checkpoint_vgg16['features.0.weight']
                model_dict['conv1.0.bias'] = checkpoint_vgg16['features.0.bias']
                model_dict['conv2.0.weight'] = checkpoint_vgg16['features.2.weight']
                model_dict['conv2.0.bias'] = checkpoint_vgg16['features.2.bias']
                model_dict['conv3.0.weight'] = checkpoint_vgg16['features.5.weight']
                model_dict['conv3.0.bias'] = checkpoint_vgg16['features.5.bias']
                model_dict['conv4.0.weight'] = checkpoint_vgg16['features.7.weight']
                model_dict['conv4.0.bias'] = checkpoint_vgg16['features.7.bias']
                model_dict['conv5.0.weight'] = checkpoint_vgg16['features.10.weight']
                model_dict['conv5.0.bias'] = checkpoint_vgg16['features.10.bias']
                model_dict['conv6.0.weight'] = checkpoint_vgg16['features.12.weight']
                model_dict['conv6.0.bias'] = checkpoint_vgg16['features.12.bias']
                model_dict['conv7.0.weight'] = checkpoint_vgg16['features.14.weight']
                model_dict['conv7.0.bias'] = checkpoint_vgg16['features.14.bias']
                model_dict['conv8.0.weight'] = checkpoint_vgg16['features.17.weight']
                model_dict['conv8.0.bias'] = checkpoint_vgg16['features.17.bias']
                model_dict['conv9.0.weight'] = checkpoint_vgg16['features.19.weight']
                model_dict['conv9.0.bias'] = checkpoint_vgg16['features.19.bias']
                model_dict['conv10.0.weight'] = checkpoint_vgg16['features.21.weight']
                model_dict['conv10.0.bias'] = checkpoint_vgg16['features.21.bias']
                model_dict['conv11.0.weight'] = checkpoint_vgg16['features.24.weight']
                model_dict['conv11.0.bias'] = checkpoint_vgg16['features.24.bias']
                model_dict['conv12.0.weight'] = checkpoint_vgg16['features.26.weight']
                model_dict['conv12.0.bias'] = checkpoint_vgg16['features.26.bias']
                model_dict['conv13.0.weight'] = checkpoint_vgg16['features.28.weight']
                model_dict['conv13.0.bias'] = checkpoint_vgg16['features.28.bias']
                encoder.load_state_dict(model_dict)

        if opt.mode.lower() == 'cluster' and not opt.vladv2:
            layers = list(encoder.children())
            layers.append(L2Norm())
            encoder = nn.Sequential(*layers)

        # encoder = nn.Sequential(*layers)
        model = nn.Module() 
        model.add_module('encoder', encoder)

        if opt.mode.lower() != 'cluster':
            if opt.pooling.lower() == 'netvlad':
                net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
                if not opt.resume: 
                    if opt.mode.lower() == 'train':
                        initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                    else:
                        initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                    if not exists(initcache):
                        logger.error('Could not find clusters, please run with --mode=cluster before proceeding')
                        raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                    with h5py.File(initcache, mode='r') as h5: 
                        clsts = h5.get("centroids")[...]
                        traindescs = h5.get("descriptors")[...]
                        net_vlad.init_params(clsts, traindescs) 
                        del clsts, traindescs

                model.add_module('pool', net_vlad)
            elif opt.pooling.lower() == 'max':
                global_pool = nn.AdaptiveMaxPool2d((1,1))
                model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
            elif opt.pooling.lower() == 'avg':
                global_pool = nn.AdaptiveAvgPool2d((1,1))
                model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
            else:
                logger.error('Unknown pooling type: {}'.format( opt.pooling))
                raise ValueError('Unknown pooling type: ' + opt.pooling)

        isParallel = False
        if opt.nGPU > 1 and torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
            if opt.mode.lower() != 'cluster':
                model.pool = nn.DataParallel(model.pool)
            isParallel = True

        if not opt.resume:
            model = model.to(device)
        
        if opt.mode.lower() == 'train':
            if opt.optim.upper() == 'ADAM':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                    model.parameters()), lr=opt.lr)#, betas=(0,0.9))
            elif opt.optim.upper() == 'SGD':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                    model.parameters()), lr=opt.lr,
                    momentum=opt.momentum,
                    weight_decay=opt.weightDecay)

                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
            else:
                logger.error('Unknown optimizer: {}'.format(opt.optim))
                raise ValueError('Unknown optimizer: ' + opt.optim)

            # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
            criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                    p=2, reduction='sum').to(device)

        if opt.resume:
            if opt.ckpt.lower() == 'latest':
                resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
            elif opt.ckpt.lower() == 'best':
                resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

            if isfile(resume_ckpt):
                # print("=> loading checkpoint '{}'".format(resume_ckpt))
                logger.info("=> loading checkpoint '{}'".format(resume_ckpt))
                checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
                opt.start_epoch = checkpoint['epoch']
                best_metric = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                model = model.to(device)
                if opt.mode == 'train':
                    optimizer.load_state_dict(checkpoint['optimizer'])
                # print("=> loaded checkpoint '{}' (epoch {})"
                #       .format(resume_ckpt, checkpoint['epoch']))
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume_ckpt, checkpoint['epoch']))
            else:
                # print("=> no checkpoint found at '{}'".format(resume_ckpt))
                logger.error("=> no checkpoint found at '{}'".format(resume_ckpt))
        
        if opt.fromtask:
            resume_ckpt = join(opt.fromtask, 'checkpoints', 'model_best.pth.tar')
            if isfile(resume_ckpt):
                # print("=> loading checkpoint '{}'".format(resume_ckpt))
                logger.info("=> loading checkpoint '{}'".format(resume_ckpt))
                checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
                pc_dict = checkpoint['pc_dict']
                for k,v in pc_dict.items():
                    pc_dict[k] = pc_dict[k].cuda() 
                source_dict = checkpoint['state_dict']
                state_dict = {k:v for k,v in source_dict.items() if 'encoder' in k}
                model_dict = model.state_dict()
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                model = model.to(device)
                logger.info("=> loaded checkpoint '{}' ".format(resume_ckpt))
            else:
                # print("=> no checkpoint found at '{}'".format(resume_ckpt))
                logger.error("=> no checkpoint found at '{}'".format(resume_ckpt))

        try:
            if opt.mode.lower() == 'test':
                # print('===> Running evaluation step')
                logger.info('===> Running evaluation step')
                epoch = 1
                recalls = test(whole_test_set, epoch, write_tboard=False)
            elif opt.mode.lower() == 'cluster':
                # print('===> Calculating descriptors and clusters')
                logger.info('===> Calculating descriptors and clusters')
                get_clusters(whole_train_set)
            elif opt.mode.lower() == 'train':
                # print('===> Training model')
                logger.info('===> Training model')
                writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

                # write checkpoints in logdir
                logdir = writer.file_writer.get_logdir()
                opt.savePath = join(logdir, opt.savePath)
                if not opt.resume:
                    makedirs(opt.savePath)

                with open(join(opt.savePath, 'flags.json'), 'w') as f:
                    f.write(json.dumps(
                        {k:v for k,v in vars(opt).items()}
                        ))
                # print('===> Saving state to:', logdir)
                logger.info('===> Saving state to: {}'.format(logdir))

                # 卷积层中OWM（正交权重修改）中的参数p
                if opt.contilearn and not opt.fromtask:
                    pc_dict = {}
                    if opt.arch == 'alexnet':
                        pc_dict['Pc1'] = torch.autograd.Variable(torch.eye(3 * 11 * 11).type(dtype), requires_grad=False)
                        pc_dict['Pc2'] = torch.autograd.Variable(torch.eye(64 * 5 * 5).type(dtype), requires_grad=False)
                        pc_dict['Pc3'] = torch.autograd.Variable(torch.eye(192 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc4'] = torch.autograd.Variable(torch.eye(384 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc5'] = torch.autograd.Variable(torch.eye(256 * 3 * 3).type(dtype), requires_grad=False)

                    elif opt.arch == 'vgg16':
                        pc_dict['Pc1'] = torch.autograd.Variable(torch.eye(3 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc2'] = torch.autograd.Variable(torch.eye(64 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc3'] = torch.autograd.Variable(torch.eye(64 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc4'] = torch.autograd.Variable(torch.eye(128 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc5'] = torch.autograd.Variable(torch.eye(128 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc6'] = torch.autograd.Variable(torch.eye(256 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc7'] = torch.autograd.Variable(torch.eye(256 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc8'] = torch.autograd.Variable(torch.eye(256 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc9'] = torch.autograd.Variable(torch.eye(512 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc10'] = torch.autograd.Variable(torch.eye(512 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc11'] = torch.autograd.Variable(torch.eye(512 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc12'] = torch.autograd.Variable(torch.eye(512 * 3 * 3).type(dtype), requires_grad=False)
                        pc_dict['Pc13'] = torch.autograd.Variable(torch.eye(512 * 3 * 3).type(dtype), requires_grad=False)
                

                not_improved = 0
                best_score = 0
                for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
                    if opt.optim.upper() == 'SGD':
                        scheduler.step(epoch)
                    train(epoch)
                    if (epoch % opt.evalEvery) == 0:
                        recalls = test(whole_test_set, epoch, write_tboard=True)
                        is_best = recalls[5] > best_score 
                        if is_best:
                            not_improved = 0
                            best_score = recalls[5]
                        else: 
                            not_improved += 1

                        save_checkpoint({
                                'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'recalls': recalls,
                                'best_score': best_score,
                                'optimizer' : optimizer.state_dict(),
                                'parallel' : isParallel,
                                'pc_dict' : pc_dict if opt.contilearn else {},
                        }, is_best)

                        if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                            logger.info('Performance did not improve for {} epochs. Stopping.'.format(opt.patience))
                            # print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                            break

                # print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
                logger.info("=> Best Recall@5: {:.4f}".format(best_score))
                writer.close()
                while True:
                    time.sleep(300)
                    logger.warning('Finished...')
        except Exception as e:
            logger.exception(e)
    except Exception as e:
        logger.exception(e)