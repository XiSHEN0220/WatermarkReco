
# coding: utf-8
import os
import sys
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../')))

from model.model import ResNetLayer3Feat, ResNetLayer4Feat

import torch
import torch.nn.functional as F

from torchvision import transforms
import numpy as np
from dataloader import TrainDataLoader
from torch.autograd import Variable
import outils
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--outDir', type=str , help='output model directory')

##---- Loss Parameter ----####

parser.add_argument(
    '--tripleLossThreshold', type=float , default = 1.0, help='threshold for triple loss')

##---- Search, Train, Validate Region ----####

parser.add_argument(
    '--searchRegion', type=int, default=1, help='feat size')



##---- Training parameters ----####


parser.add_argument(
    '--modelPth', type=str, default = '../model/net98_8.pth', help='finetune model weight path')

parser.add_argument(
    '--searchDir', type=str, default= '../data/watermark/C_cross_domain/trainBG2/', help='searching directory')
    
parser.add_argument(
    '--queryDir', type=str, default= '../data/watermark/C_cross_domain/trainBG2/', help='query image directory')


parser.add_argument(
    '--nbEpoch', type=int , default = 300, help='Number of training epochs')

parser.add_argument(
    '--lr', type=float , default = 1e-5, help='learning rate')


parser.add_argument(
    '--batchSize', type=int , default = 8, help='batch size')

parser.add_argument(
    '--cuda', action='store_true', help='cuda setting')

parser.add_argument(
    '--nbSearchImgEpoch', type=int, default = 2000, help='maximum number of searching image in one epoch')

parser.add_argument(
    '--featScaleBase', type=int, default= 22, help='median # of features in the scale list ')

parser.add_argument(
    '--stepNbFeat', type=int, default= 3, help='difference nb feature in adjacent scales ')

parser.add_argument(
    '--nbscale', type=int, default= 2, help='# of octaves')



    
parser.add_argument(
    '--featLayer', type = str, default='conv4', choices=['conv4', 'conv5'], help='which feature, conv4 or conv5')

parser.add_argument(
    '--labelInfo', type = str, default='../data/crossDomainTraining.json', help='label category')

parser.add_argument(
    '--tmpTrainDir', type = str, default='./trainPair', help='temporal image directory to store training pairs')

parser.add_argument(
    '--eta', type = float, default=1e-7, help='eta for calculate norm')

parser.add_argument(
    '--margin', type = int, default=3, help='keep top K ')
    
parser.add_argument(
    '--tolerance', type = float, default=4, help='tolerance')

parser.add_argument(
    '--valK', type = int, default=300, help='keep top K for validation')


parser.add_argument(
    '--K', type = int, default=300, help='keep top K ')

parser.add_argument(
    '--dataset', type = str, default='watermark', choices = ['watermark', 'sketch'], help='running on which dataset')



args = parser.parse_args()
tqdm.monitor_interval = 0
print (args)

if os.path.exists(args.tmpTrainDir) : 
    cmd = 'rm -r {}'.format(args.tmpTrainDir)
    os.system(cmd)
    
os.mkdir(args.tmpTrainDir)

## Dataset, Minimum dimension, Total patch during the training
with open(args.labelInfo, 'r') as f : 
    label = json.load(f)
    
QueryImgList = sorted(label['queryImg'])
SearchImgList = sorted(label['searchImg'])
labelCategory = label['annotation']


nbPatchTotal = args.nbSearchImgEpoch
imgFeatMin = args.searchRegion + 2 * args.margin + 1 ## Minimum dimension of feature map in a image
tolerance = args.tolerance / float(args.featScaleBase)

## Loading model


if args.dataset == 'watermark': 
    normalize = transforms.Normalize(mean = [ 0.75, 0.70, 0.65 ], std = [ 0.14,0.15,0.16 ]) ## watermark classification normalization
    if args.featLayer == 'conv4' : 
        net = ResNetLayer3Feat( None )
    
        msg = 'loading weight from {}'.format(args.modelPth)
        print (msg)
        modelParams = torch.load(args.modelPth)
        for key in list(modelParams.keys()) : 
            if 'layer4' in key  or 'fc.fc1' in key: 
                modelParams.pop(key, None)
        
    
    else : 
        net = ResNetLayer4Feat( None )
        msg = 'loading weight from {}'.format(args.modelPth)
        print (msg)
        for key in list(modelParams.keys()) : 
            if 'fc.fc1' in key: 
                modelParams.pop(key, None)
        modelParams = torch.load(args.modelPth)
    net.load_state_dict( modelParams )
else : 
    normalize = transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]) ## imagenet classification normalization
    if args.featLayer == 'conv4' : 
        net = ResNetLayer3Feat( None )
    
        msg = 'loading weight from {}'.format(args.modelPth)
        print (msg)
        modelParams = torch.load(args.modelPth)
        for key in list(modelParams.keys()) : 
            if 'layer4' in key  or 'fc' in key: 
                modelParams.pop(key, None)
        
    
    else : 
        net = ResNetLayer4Feat( None )
        msg = 'loading weight from {}'.format(args.modelPth)
        print (msg)
        for key in list(modelParams.keys()) : 
            if 'fc' in key: 
                modelParams.pop(key, None)
        modelParams = torch.load(args.modelPth)

    net.load_state_dict( modelParams )
    
if args.cuda : 
    net.cuda()

featChannel = 256 if args.featLayer == 'conv4' else 512
## stride size, min input size, channel of feature 
strideNet = 16
minNet = 15
PATIENCE = 10 # 10 epochs no improved, training will be stopped
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))

transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])
                                
## Scales
scales = [args.featScaleBase - args.stepNbFeat * i for i in range(args.nbscale, 0, -1)] + [args.featScaleBase] + [args.featScaleBase + args.stepNbFeat * i for i in range(1, args.nbscale + 1)]
msg = 'We search to match in {:d} scales, the max dimensions in the feature maps are:'.format(len(scales))
print (msg)
print (scales)
print ('\n\n')



## Output
if not os.path.exists(args.outDir) :
    os.mkdir(args.outDir)
history = {'posLoss':[], 'negaLoss':[], 'nbPos':[]}
nbEpochNoImproved = 0
bestNbPos = 0
outHistory = os.path.join(args.outDir, 'history.json')

if len(QueryImgList) <= args.nbSearchImgEpoch :
    valQueryImgList = QueryImgList
    valSearchImgList = SearchImgList
else :
    index = np.linspace(0, len(QueryImgList)-1, args.nbSearchImgEpoch).astype(np.int64)
    valQueryImgList = [QueryImgList[i] for i in index]
    valSearchImgList = [SearchImgList[i] for i in index]
    
    
        
## Main Loop
for i_ in range(args.nbEpoch) :
    logPosLoss = []
    logNegaLoss = []

    print ('Training Epoch {:d}'.format(i_))
    print ('---> Get query...')
    net.eval()
    if len(QueryImgList) <= args.nbSearchImgEpoch :
        queryImgList = QueryImgList
    else :
        index = np.random.permutation(np.arange(len(QueryImgList)))[:args.nbSearchImgEpoch]
        queryImgList = [QueryImgList[i] for i in index]
        
    if len(SearchImgList) <= args.nbSearchImgEpoch :
        searchImgList = SearchImgList
    else :
        index = np.random.permutation(np.arange(len(SearchImgList)))[:args.nbSearchImgEpoch]
        searchImgList = [SearchImgList[i] for i in index]

    featQuery, queryNameList = outils.RandomQueryFeat(nbPatchTotal, featChannel, args.searchRegion, imgFeatMin, minNet, strideNet, transform, net, args.searchDir, args.margin, queryImgList, args.cuda, args.featScaleBase)

    print ('---> Get topK patches matching to query...')
    topkImg, topkScale, topkValue, topkW, topkH = outils.RetrievalRes(nbPatchTotal, searchImgList, args.searchDir, args.margin, args.searchRegion, scales, minNet, strideNet, transform, net, featQuery, args.cuda, min(len(searchImgList), args.K))

    print ('---> Get training pairs...')
    posPair, negPair = outils.TrainPair(args.searchDir, searchImgList, topkImg, topkScale, topkW, topkH, transform, net, args.margin, args.cuda, featChannel, args.searchRegion, minNet, strideNet, labelCategory, min(len(searchImgList), args.K), queryNameList, tolerance)


    outils.saveTrainImgPair(posPair, negPair, args.tmpTrainDir, args.margin, args.searchDir, searchImgList, topkImg, topkScale, topkW, topkH, minNet, strideNet, args.searchRegion)
    
    msg = 'NB Pos Train Pair : {:d}, NB Neg Train Pair : {:d}'.format(len(posPair), len(negPair))
    print (msg)
    if len(posPair) < args.batchSize : 
        continue
    trainloader = TrainDataLoader(args.tmpTrainDir, len(posPair), transform, args.batchSize)

    ## Calculate Loss
    net.train() # switch to train mode
    net.trainFreezeBN()
    for batch in trainloader :
        p1, p2, n1, n2 = batch['posI1'], batch['posI2'], batch['negI1'], batch['negI2']
        if args.cuda :
            p1, p2, n1, n2 = p1.cuda(), p2.cuda(), n1.cuda(), n2.cuda()
        
        optimizer.zero_grad()
        p1, p2, n1, n2 = net(p1), net(p2), net(n1), net(n2)
        posSimilarityBatch, negaSimilarityBatch = outils.CosSimilarity(p1, p2, n1, n2, args.margin, args.eta)
        
        ## Triplet Loss
        loss = torch.clamp(negaSimilarityBatch  + args.tripleLossThreshold - 1, min=0) + torch.clamp(args.tripleLossThreshold - posSimilarityBatch, min=0)
        ## make sure that gradient is not zero
        if (loss > 0).any() :
            loss = loss.mean()
            loss.backward()
            optimizer.step()

        logPosLoss.append( posSimilarityBatch.mean().item() )
        logNegaLoss.append( negaSimilarityBatch.mean().item() )

    # Save model, training history; print loss
    msg = 'EPOCH {:d}, positive pairs similarity: {:.4f}, negative pairs similarity: {:.4f}'.format(i_, np.mean(logPosLoss), np.mean(logNegaLoss))
    print (msg)
    history['posLoss'].append(np.mean(logPosLoss))
    history['negaLoss'].append(np.mean(logNegaLoss))

    ## VALIDATION 
    net.eval()
    featQuery, QueryImgList = outils.RandomQueryFeat(nbPatchTotal, featChannel, args.searchRegion, imgFeatMin, minNet, strideNet, transform, net, args.searchDir, args.margin, valQueryImgList, args.cuda, args.featScaleBase, False)

    print ('---> Get top10 patches matching to query...')
    topkImg, topkScale, topkValue, topkW, topkH = outils.RetrievalRes(nbPatchTotal, valSearchImgList, args.searchDir, args.margin, args.searchRegion, scales, minNet, strideNet, transform, net, featQuery, args.cuda, min(len(valSearchImgList), args.valK))

    print ('---> Get validation pairs...')
    
    posPair, _ = outils.TrainPair(args.searchDir, valSearchImgList, topkImg, topkScale, topkW, topkH, transform, net, args.margin, args.cuda, featChannel, args.searchRegion, minNet, strideNet, labelCategory, min(len(SearchImgList), args.valK), QueryImgList, tolerance, False)
    history['nbPos'].append(len(posPair))
    msg = 'VALIDATION : Number of spatial consistency pairs : {:d}'.format(len(posPair))
    print (msg)
    with open(outHistory, 'w') as f :
        json.dump(history, f)
        
    if len(posPair) > bestNbPos:
        outModelPath = os.path.join(args.outDir, 'net.pth') 
        msg = 'VALIDATION IMPROVED, Number of spatial consistency pairs improved from {:d} to {:d} \n Model will be saved into {}...'.format(bestNbPos, len(posPair), outModelPath)
        print (msg)
        bestNbPos = len(posPair)
        torch.save(net.state_dict(), outModelPath)
        nbEpochNoImproved = 0
    else : 
        nbEpochNoImproved += 1
    if nbEpochNoImproved == PATIENCE :
        msg = '{:d} Epochs no improved, training stop'.format(nbEpochNoImproved) 
        print (msg)
        break
    
