import os 
import sys
from tqdm import tqdm
import argparse
from torch import nn
import torch
from torch.autograd import Variable
import PIL.Image as Image
from shutil import copyfile
import cv2
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../')))

from model.model import ResNetLayer3Feat, ResNetLayer4Feat
from torchvision import datasets, transforms,models
import json 
import torch.nn.functional as F
import numpy as np
from scipy.misc import imresize
parser = argparse.ArgumentParser()


##---- Model Setting ----####

parser.add_argument(
    '--modelPth', type=str, default='../model/net98_8.pth', help='model weight path')

parser.add_argument(
    '--labelJson', type=str, default = '../data/BTestBriquet.json', help='labels json file')

parser.add_argument(
    '--searchDir', type=str, default = '../data/watermark/briquet_synthetic/', help='searching image dataset')

parser.add_argument(
    '--queryDir', type=str, default = '../data/watermark/B_cross_domain_plus/val/', help='query image dataset')

parser.add_argument(
    '--outResJson', type=str, default = 'res.json', help='output json file to store the results')
    

parser.add_argument(
    '--imgScale', type = int, default=14, help='input image size (# feat in conv4), 14 means input image size is 224 * 224 ')
    
parser.add_argument(
    '--nbSaveRefFeat', type = int, default= 1000, help='Nb of store feature')



args = parser.parse_args()
print ('\n\n\n')
print (args)

def getFeat(I, net, transform) : 
    feat = net(transform(I).unsqueeze(0).cuda())
    feat = feat.data.squeeze()
    feat = feat.view(feat.size(0), -1).mean(1)
    feat = F.normalize(feat, dim=0, p=2)
    
    return feat
    
normalize = transforms.Normalize(mean = [ 0.75, 0.70, 0.65 ], std = [ 0.14,0.15,0.16 ]) ## watermark classification normalization    
net = ResNetLayer3Feat( None )

msg = 'loading weight from {}'.format(args.modelPth)
print (msg)
modelParams = torch.load(args.modelPth)
for key in list(modelParams.keys()) : 
    if 'layer4' in key  or 'fc.fc1' in key: 
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

res = [[] for i in range(len(label['val']))]

splitSearchImgList = []
for i in range(len(label['searchImg']) // args.nbSaveRefFeat + 1)  : 
    splitSearchImgList.append(label['searchImg'][i * args.nbSaveRefFeat : min((i + 1) * args.nbSaveRefFeat, len(label['searchImg']))])
    
nbGroup = len(splitSearchImgList)

FlipRotation = [[False, 0], [False, 90], [False, 180], [False, 270], [True, 0], [True, 90], [True, 180], [True, 270]]
with torch.no_grad() : 
    RefFeat = torch.cuda.FloatTensor(args.nbSaveRefFeat, 8, 256).zero_()
    QueryFeat = torch.cuda.FloatTensor(1, 1, 256).zero_()

    
    for i in tqdm(range(nbGroup)) : 
        RefFeat = RefFeat.zero_()
          
        for t, targetImgName in tqdm(enumerate(splitSearchImgList[i])) : 
            targetImgPath = os.path.join(args.searchDir, targetImgName)
            I = Image.open(targetImgPath).convert('RGB')
            
            
            RefFeat[t][0] =  getFeat(I, net, transform)
            RefFeat[t][1] =  getFeat(I.rotate(90), net, transform)
            RefFeat[t][2] =  getFeat(I.rotate(180), net, transform)
            RefFeat[t][3] =  getFeat(I.rotate(270), net, transform)
            
            Iflip = I.transpose(Image.FLIP_LEFT_RIGHT)
            
            RefFeat[t][4] =  getFeat(Iflip, net, transform)
            RefFeat[t][5] =  getFeat(Iflip.rotate(90), net, transform)
            RefFeat[t][6] =  getFeat(Iflip.rotate(180), net, transform)
            RefFeat[t][7] =  getFeat(Iflip.rotate(270), net, transform)
            

            
        for j, sourceImgName in tqdm(enumerate(label['val'])) :
            sourceImgPath = os.path.join(args.queryDir, sourceImgName)
            I = Image.open(sourceImgPath).convert('RGB')
            QueryFeat[0, 0] = getFeat(I, net, transform)
            score = torch.sum(RefFeat * QueryFeat, dim=2) # nbSaveRefFeat * 8
            value, indexRot = torch.max( score, dim=1)
            value, indexImg = torch.sort(value, dim=0, descending=True)
            
            res[j] =  res[j] + [[value[k].item(), splitSearchImgList[i][indexImg[k].item()]] + FlipRotation[indexRot[indexImg[k]]] for k in range(len(splitSearchImgList[i]))]

res = [sorted(res[i], key=lambda s: s[0], reverse=True) for i in range(len(res))]

with open(args.outResJson, 'w') as f : 
    json.dump(res, f)
    
nbSourceImg = len(res)
truePosCount = 0
truePosTop10Count = 0
truePosTop100Count = 0
truePosTop1000Count = 0


for i in range(len(res)) : 
    sourceImg = label['val'][i]
    category = label['annotation'][sourceImg]
    top1000 = [label['annotation'][res[i][j][1]] for j in range(1000)]
    truePosTop1000Count = truePosTop1000Count + 1 if category in top1000 else truePosTop1000Count
    truePosTop100Count = truePosTop100Count + 1 if category in top1000[:100] else truePosTop100Count
    truePosTop10Count = truePosTop10Count + 1 if category in top1000[:10] else truePosTop10Count
    truePosCount = truePosCount + 1 if category == top1000[0] else truePosCount
    

msg = '***** Top1 is {:.3f}, Top 10 {:.3f}, Top 100 {:.3}, Top1000 {:.3f}*****'.format(
                                                                                        truePosCount / nbSourceImg, 
                                                                                        truePosTop10Count / nbSourceImg,
                                                                                        truePosTop100Count / nbSourceImg,
                                                                                        truePosTop1000Count / nbSourceImg,
                                                                                        )
print (msg)
