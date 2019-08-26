import pair_discovery 
import outils
import os 
import sys
from tqdm import tqdm
import argparse
import torch
import numpy as np
from shutil import copyfile
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../')))

from model.model import ResNetLayer3Feat, ResNetLayer4Feat

from torchvision import transforms
import json 
import PIL.Image as Image 
import time            
            
parser = argparse.ArgumentParser()

##---- Search Dataset Setting ----####
parser.add_argument(
    '--featScaleBase', type=int, default= 22, help='median # of features in the scale list ')

parser.add_argument(
    '--stepNbFeat', type=int, default= 3, help='difference nb feature in adjacent scales ')

parser.add_argument(
    '--nbscale', type=int, default= 2, help='# of octaves')


##---- Model Setting ----####

parser.add_argument(
    '--modelPth', type=str, default='../model/net98_8.pth', help='model weight path')

parser.add_argument(
    '--margin', type=int, default= 3, help='margin, the feature describing the border part is not taken into account')

parser.add_argument(
    '--tolerance', type=float , default = 2., help='tolerance expressed by nb of features (2 for retrieval with image 1 for retrieval with region)')

parser.add_argument(
    '--labelJson', type=str, default = '../data/BTestBriquet.json', help='labels json file')
    
parser.add_argument(
    '--preOrderJson', type=str, default = '../data/LocalSimiFinetune.json', help='labels json file')

parser.add_argument(
    '--evaluateTopK', type=int, default = 100, help='Evaluating Top K references in the preOrder Json file ')

parser.add_argument(
    '--searchDir', type=str, default = '../data/watermark/briquet_synthetic/', help='searching image dataset')

parser.add_argument(
    '--queryDir', type=str, default = '../data/watermark/B_cross_domain_plus/val/', help='query image dataset')

parser.add_argument(
    '--outJson', type=str, help='output json file to store the results')
    
parser.add_argument(
    '--eta', type = float, default=1e-7, help='eta to compute norm')

parser.add_argument(
    '--scoreType', type = str, default='Identity', choices = ['Identity', 'Hough', 'Affine'], help='type of score')

parser.add_argument(
    '--indexBegin', type = int, default=0, help='begin index')

parser.add_argument(
    '--indexEnd', type = int, default=200, help='end index')



args = parser.parse_args()
print ('\n\n\n')
print (args)


start = time.time()
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
strideNet = 16
minNet = 15
    
scaleList = [args.featScaleBase - args.stepNbFeat * i for i in range(args.nbscale, 0, -1)] + [args.featScaleBase] + [args.featScaleBase + args.stepNbFeat * i for i in range(1, args.nbscale + 1)]
print (scaleList)

with open(args.labelJson, 'r') as f :
    label = json.load(f)
    
with open(args.preOrderJson, 'r') as f :
    preOrder = json.load(f)

index = [i for i in range(args.indexBegin, args.indexEnd)]

transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])
                                

res = []
res = [[] for i in index]
with torch.no_grad() : 
    for i in tqdm(range(len(index))) :
        sourceImgName = label['val'][index[i]]
        sourceImgPath = os.path.join(args.queryDir, sourceImgName)
        
        for j in tqdm(range(args.evaluateTopK)) :
            _, targetImgName, flip, rotation = preOrder[index[i]][j]
            
            RefFeat = {targetImgName:[]}
            targetImgPath = os.path.join(args.searchDir, targetImgName)
            I = Image.open(targetImgPath).convert('RGB')
                
            I = I.transpose(Image.FLIP_LEFT_RIGHT) if flip else I
            I = I.rotate(rotation)
            for scale in scaleList :
                RefFeat[targetImgName].append(outils.imgFeat(minNet, strideNet, I, net, transform, scale, args.eta))
             
            score, inlier = pair_discovery.PairDiscovery(sourceImgName, args.queryDir, targetImgName, args.searchDir, net, transform, args.tolerance, args.margin, args.featScaleBase, scaleList, args.eta, 'conv4', args.scoreType, RefFeat, False)
            res[i].append((score, targetImgName, flip, rotation))
        
        
res = [sorted(res[i], key=lambda s: s[0], reverse=True) for i in range(len(res))]

with open(args.outJson, 'w') as f : 
    json.dump(res, f)

nbSourceImg = len(res)
truePosCount = 0
truePosTop2Count = 0
truePosTop3Count = 0
truePosTop5Count = 0
truePosTop10Count = 0
truePosTop20Count = 0
truePosTop40Count = 0
truePosTop50Count = 0
truePosTop100Count = 0
truePosTop1000Count = 0




for i in range(len(res)) : 
    sourceImg = label['val'][index[i]]
    category = label['annotation'][sourceImg]
    top50 = [label['annotation'][res[i][j][1]] for j in range(min(1000, args.evaluateTopK))]
    truePosTop1000Count = truePosTop1000Count + 1 if category in top50[:min(1000, args.evaluateTopK)] else truePosTop1000Count
    truePosTop100Count = truePosTop100Count + 1 if category in top50[:100] else truePosTop100Count
    truePosTop50Count = truePosTop50Count + 1 if category in top50[:50] else truePosTop50Count
    truePosTop40Count = truePosTop40Count + 1 if category in top50[:40] else truePosTop40Count
    truePosTop20Count = truePosTop20Count + 1 if category in top50[:20] else truePosTop20Count
    truePosTop10Count = truePosTop10Count + 1 if category in top50[:10] else truePosTop10Count
    truePosTop5Count = truePosTop5Count + 1 if category in top50[:5] else truePosTop5Count
    truePosTop3Count = truePosTop3Count + 1 if category in top50[:3] else truePosTop3Count
    truePosTop2Count = truePosTop2Count + 1 if category in top50[:2] else truePosTop2Count
    truePosCount = truePosCount + 1 if category == top50[0] else truePosCount
    

msg = "***** Time {:.3f}s, Top1 is {:.3f}, Top2 is {:.3f}, \
       Top3 is {:.3f}, Top5 is {:.3f}, \
       Top10 is {:.3f}, Top 20 is {:.3f}, \
       Top 40 is {:.3}, Top50 is {:.3f}, Top100 is {:.3f}, Top1000 is {:.3f}*****".format(  time.time() - start,
                                                                                             truePosCount / nbSourceImg,
                                                                                             truePosTop2Count / nbSourceImg,
                                                                                             truePosTop3Count / nbSourceImg,
                                                                                             truePosTop5Count / nbSourceImg,
                                                                                             truePosTop10Count / nbSourceImg,
                                                                                             truePosTop20Count / nbSourceImg,
                                                                                             truePosTop40Count / nbSourceImg,
                                                                                             truePosTop50Count / nbSourceImg,
                                                                                             truePosTop100Count / nbSourceImg,
                                                                                             truePosTop1000Count / nbSourceImg)
                                                     
                                                     
                                                     
print (msg)
                                                     
                                                      













nbSourceImg = len(res)
truePosCount = 0
truePosTop2Count = 0
truePosTop3Count = 0
truePosTop5Count = 0
truePosTop10Count = 0
truePosTop20Count = 0
truePosTop40Count = 0
truePosTop50Count = 0
truePosTop100Count = 0
truePosTop1000Count = 0




for i in range(len(res)) : 
    sourceImg = label['val'][index[i]]
    category = label['annotation'][sourceImg]
    top50 = [label['annotation'][preOrder[index[i]][j][1]] for j in range(1000)]
    truePosTop1000Count = truePosTop1000Count + 1 if category in top50[:1000] else truePosTop1000Count
    truePosTop100Count = truePosTop100Count + 1 if category in top50[:100] else truePosTop100Count
    truePosTop50Count = truePosTop50Count + 1 if category in top50[:50] else truePosTop50Count
    truePosTop40Count = truePosTop40Count + 1 if category in top50[:40] else truePosTop40Count
    truePosTop20Count = truePosTop20Count + 1 if category in top50[:20] else truePosTop20Count
    truePosTop10Count = truePosTop10Count + 1 if category in top50[:10] else truePosTop10Count
    truePosTop5Count = truePosTop5Count + 1 if category in top50[:5] else truePosTop5Count
    truePosTop3Count = truePosTop3Count + 1 if category in top50[:3] else truePosTop3Count
    truePosTop2Count = truePosTop2Count + 1 if category in top50[:2] else truePosTop2Count
    truePosCount = truePosCount + 1 if category == top50[0] else truePosCount
    

msg = "***** Time {:.3f}s, Top1 is {:.3f}, Top2 is {:.3f}, \
       Top3 is {:.3f}, Top5 is {:.3f}, \
       Top10 is {:.3f}, Top 20 is {:.3f}, \
       Top 40 is {:.3}, Top50 is {:.3f}, Top100 is {:.3f}, Top1000 is {:.3f}*****".format(  time.time() - start,
                                                                                             truePosCount / nbSourceImg,
                                                                                             truePosTop2Count / nbSourceImg,
                                                                                             truePosTop3Count / nbSourceImg,
                                                                                             truePosTop5Count / nbSourceImg,
                                                                                             truePosTop10Count / nbSourceImg,
                                                                                             truePosTop20Count / nbSourceImg,
                                                                                             truePosTop40Count / nbSourceImg,
                                                                                             truePosTop50Count / nbSourceImg,
                                                                                             truePosTop100Count / nbSourceImg,
                                                                                             truePosTop1000Count / nbSourceImg)
                                                     
                                                     
                                                     
print (msg)
                                                     
                                                      


