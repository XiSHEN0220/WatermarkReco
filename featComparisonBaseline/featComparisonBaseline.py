import os 
import sys
from tqdm import tqdm
import argparse
from torch import nn
import torch
from torch.autograd import Variable
import PIL.Image as Image
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../')))

from model.model import ResNetLayer3Feat, ResNetLayer4Feat
from torchvision import datasets, transforms,models
import json 
from shutil import copyfile
import cv2
import numpy as np
from scipy.misc import imresize

def getFeat(I, net, transform) : 
    Feat = net(transform(I).unsqueeze(0).cuda())
    Feat = Feat.data
    
    return Feat
    
    
def AvgPool(sourceImgFeat, targetImgFeat) : 

    sourceImgFeat = sourceImgFeat.mean(dim=2).mean(dim=2).squeeze()
    sourceImgFeat = sourceImgFeat / torch.sum( sourceImgFeat ** 2 ) ** 0.5
    
    targetImgFeat = targetImgFeat.mean(dim=2).mean(dim=2).squeeze()
    targetImgFeat = targetImgFeat / torch.sum( targetImgFeat ** 2 ) ** 0.5
    
    score = torch.sum(sourceImgFeat * targetImgFeat)
    
    return score.item()
    
def LocalSimi(sourceImgFeat, targetImgFeat) : 

    sourceImgFeat = sourceImgFeat / torch.sum( sourceImgFeat ** 2, dim = 1, keepdim=True) ** 0.5
    
    targetImgFeat = targetImgFeat / torch.sum( targetImgFeat ** 2, dim = 1, keepdim=True) ** 0.5
    
    score = torch.sum(sourceImgFeat * targetImgFeat, dim=1)
    score = torch.mean(score)
    
    return score.item()

def Cat(sourceImgFeat, targetImgFeat) : 

    sourceImgFeat = sourceImgFeat / torch.sum( sourceImgFeat ** 2) ** 0.5
    
    targetImgFeat = targetImgFeat / torch.sum( targetImgFeat ** 2) ** 0.5
    
    score = torch.sum(sourceImgFeat * targetImgFeat) # W  * H
    
    return score.item()
    
    
##---- Model Setting ----####
parser = argparse.ArgumentParser()

parser.add_argument(
    '--modelPth', type=str, default='../model/resnet18.pth', help='model weight path')

parser.add_argument(
    '--labelJson', type=str, default = '../data/labelOneshot.json', help='labels json file')

parser.add_argument(
    '--searchDir', type=str, default = '../data/watermark/B_one_shot_classification/', help='searching image dataset')

parser.add_argument(
    '--queryDir', type=str, default = '../data/watermark/B_one_shot_classification/', help='query image dataset')

parser.add_argument(
    '--outResJson', type=str, default = 'res.json', help='output json file to store the results')
    
parser.add_argument(
    '--featLayer', type = str, default='conv4', choices=['conv4', 'conv5'], help='which feature, conv4 or conv5')

parser.add_argument(
    '--imgScale', type = int, default=14, help='input image size (# feat in conv4), 14 means input image size is 224 * 224 ')
    
parser.add_argument(
    '--flip', action='store_true', help='Horizontal and Vertical Flip????')

parser.add_argument(
    '--saveRefFeat', action='store_true', help='For small number ref images, save features can largely reduce running time')


parser.add_argument(
    '--dataset', type = str, default='watermark', choices = ['watermark', 'sketch'], help='running on which dataset')
    

parser.add_argument(
    '--featName', type = str, default='AvgPool', choices = ['AvgPool', 'Cat', 'LocalSimi'], help='Which baseline?')

args = parser.parse_args()
print ('\n\n\n')
print (args)

if args.featName == 'AvgPool' : 
    scoreFunc = AvgPool
elif args.featName == 'Cat' : 
    scoreFunc = Cat
elif args.featName == 'LocalSimi' : 
    scoreFunc = LocalSimi
    
    
    
msg = 'loading weight from {}'.format(args.modelPth)
print (msg)
modelParams = torch.load(args.modelPth)
for key in list(modelParams.keys()) : 
    if 'fc' in key: 
        modelParams.pop(key, None)

## Define normalization
normalize = transforms.Normalize(mean = [ 0.75, 0.70, 0.65 ], std = [ 0.14,0.15,0.16 ]) if args.dataset == 'watermark' else transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])

## Architecture
net = ResNetLayer3Feat( None ) if args.featLayer == 'conv4' else ResNetLayer4Feat( None )

## if conv4 
if args.featLayer == 'conv4': 
    for key in list(modelParams.keys()) : 
        if 'layer4' in key: 
            modelParams.pop(key, None) 
        
        
net.load_state_dict( modelParams )
net.eval()
net.cuda()

transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Resize(args.imgScale * 16),
                                transforms.ToTensor(),
                                normalize,
                            ])

    

    
with open(args.labelJson, 'r') as f :
    label = json.load(f)

if args.saveRefFeat : 
    RefFeat = {}
    for targetImgName in tqdm(label['searchImg']) : 
        targetImgPath = os.path.join(args.searchDir, targetImgName)
        I = Image.open(targetImgPath).convert('RGB')
        with torch.no_grad() : 
            RefFeat[targetImgName] = [getFeat(I, net, transform)]
            if args.flip : 
                IHflip = I.transpose(Image.FLIP_LEFT_RIGHT)
                RefFeat[targetImgName].append(getFeat(IHflip, net, transform))
            
         
with torch.no_grad() : 
    res = {}
    for sourceImgName in tqdm(label['val']) :
        sourceImgPath = os.path.join(args.queryDir, sourceImgName)
        res[sourceImgName] = []
        Is = Image.open(sourceImgPath).convert('RGB')
        sourceImgFeat = getFeat(Is, net, transform)
        
        for targetImgName in tqdm(label['searchImg']) : 
            if args.saveRefFeat : 
                targetImgFeat = RefFeat[targetImgName][0]
            else :  
                targetImgPath = os.path.join(args.searchDir, targetImgName)
                ## Normalization
                I = Image.open(targetImgPath).convert('RGB')
                targetImgFeat = getFeat(I, net, transform) 
            
            score = scoreFunc(sourceImgFeat, targetImgFeat)
            flip = False
            
            
            if args.flip : 
                if args.saveRefFeat : 
                    targetImgFeat = RefFeat[targetImgName][1]
                else : 
                    IHflip = I.transpose(Image.FLIP_LEFT_RIGHT)
                    targetImgFeat = getFeat(IHflip, net, transform) 
                    
                scoreFlip = scoreFunc(sourceImgFeat, targetImgFeat)
                if scoreFlip > score : 
                    score, flip = scoreFlip, True
        
            res[sourceImgName].append((targetImgName, score, flip, 0))
    
        
nbSourceImg = len(res.keys())
truePosCount = 0
truePosTop10Count = 0

for sourceImgName in res.keys() : 
    res[sourceImgName] = sorted(res[sourceImgName], key=lambda s: s[1], reverse=True)
    predTop10 = []
    for i in range(len(res[sourceImgName])) : 
        if len(predTop10) < 10 and label['annotation'][res[sourceImgName][i][0]] not in predTop10 : 
            predTop10.append(label['annotation'][res[sourceImgName][i][0]])
            
    if label['annotation'][sourceImgName] == predTop10[0] : 
        truePosCount += 1
    if label['annotation'][sourceImgName] in predTop10 : 
        truePosTop10Count += 1
        
msg = '***** Final accuracy is {:.3f}, Top 10 {:.3f}*****'.format(truePosCount / float(nbSourceImg), truePosTop10Count / nbSourceImg)
print (msg)

with open(args.outResJson, 'w') as f : 
    json.dump(res, f)
