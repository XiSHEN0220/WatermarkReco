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
            
            
parser = argparse.ArgumentParser()

##---- Search Dataset Setting ----####
parser.add_argument(
    '--featScaleBase', type=int, default= 11, help='minimum # of features in the scale list ')

parser.add_argument(
    '--scalePerOctave', type=int, default= 3, help='# of scales in one octave ')

parser.add_argument(
    '--nbOctave', type=int, default= 2, help='# of octaves')

##---- Model Setting ----####

parser.add_argument(
    '--modelPth', type=str, default='../model/net98_8.pth', help='model weight path')

parser.add_argument(
    '--margin', type=int, default= 3, help='margin, the feature describing the border part is not taken into account')

parser.add_argument(
    '--tolerance', type=float , default = 2., help='tolerance expressed by nb of features (2 for retrieval with image 1 for retrieval with region)')

parser.add_argument(
    '--scaleImgRef', type=int , default = 22, help='maximum feature in the target image')

parser.add_argument(
    '--labelJson', type=str, default = '../data/labelBOneshotVal.json', help='labels of cross domain matching')

parser.add_argument(
    '--searchDir', type=str, default = '../data/watermark/B_cross_domain/valBG2/', help='searching image dataset')

parser.add_argument(
    '--queryDir', type=str, default = '../data/watermark/B_cross_domain/valBG2/', help='query image dataset')

parser.add_argument(
    '--outDir', type=str, help='output json file to store the results')
    
parser.add_argument(
    '--flip', action='store_true', help='Horizontal Flip????')

parser.add_argument(
    '--saveRefFeat', action='store_true', help='For small number ref images, save features can largely reduce running time')


parser.add_argument(
    '--featLayer', type = str, default='conv4', choices=['conv4', 'conv5'], help='which feature, conv4 or conv5')

parser.add_argument(
    '--eta', type = float, default=1e-7, help='eta to compute norm')

parser.add_argument(
    '--scoreType', type = str, default='Identity', choices = ['Identity', 'Hough', 'Affine'], help='type of score')

parser.add_argument(
    '--dataset', type = str, default='watermark', choices = ['watermark', 'sketch'], help='running on which dataset')


args = parser.parse_args()
print ('\n\n\n')
print (args)
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

net.eval()
net.cuda()
strideNet = 16
minNet = 15
	
scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)


with open(args.labelJson, 'r') as f :
    label = json.load(f)

transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                                ])
                                

res = {}

RefFeat = {}
RefFeatFlip = {}
if args.saveRefFeat : 
    for targetImgName in tqdm(label['searchImg']) : 
        RefFeat[targetImgName] = []
        RefFeatFlip[targetImgName] = []
        targetImgPath = os.path.join(args.searchDir, targetImgName)
        I = Image.open(targetImgPath).convert('RGB')
        IFlip = I.transpose(Image.FLIP_LEFT_RIGHT)
        with torch.no_grad() : 
            for scale in scaleList : 
                RefFeat[targetImgName].append(outils.imgFeat(minNet, strideNet, I, net, transform, scale, args.eta))
                RefFeatFlip[targetImgName].append(outils.imgFeat(minNet, strideNet, IFlip, net, transform, scale, args.eta))
                
                    
for sourceImgName in tqdm(label['val']) :
    sourceImgPath = os.path.join(args.queryDir, sourceImgName)
    res[sourceImgName] = []
    for targetImgName in tqdm(label['searchImg']) : 
        targetImgPath = os.path.join(args.searchDir, targetImgName)
        maxScore, bestInlier, flip = 0, {}, False ## to find best image among flipped image
        
        
        score, inlier = pair_discovery.PairDiscovery(sourceImgName, args.queryDir, targetImgName, args.searchDir, net, transform, args.tolerance, args.margin, args.scaleImgRef, scaleList, args.eta, args.featLayer, args.scoreType, RefFeat, False)
        
        if score > maxScore : 
            maxScore, bestInlier, flipBest = score, inlier, False
        
        
        
        if args.flip : 
            scoreflipH, inlier = pair_discovery.PairDiscovery(sourceImgName, args.queryDir, targetImgName, args.searchDir, net, transform, args.tolerance, args.margin, args.scaleImgRef, scaleList, args.eta, args.featLayer, args.scoreType, RefFeatFlip, True)
        else : 
            scoreflipH, inlier = 0, 0
        
        if scoreflipH > maxScore : 
            maxScore, bestInlier, flipBest = scoreflipH, inlier, True 
        
        
        
        res[sourceImgName].append((targetImgName, maxScore, bestInlier, flipBest))
    
    
    

nbSourceImg = len(list(res.keys()))
truePosCount = 0
truePosTop10Count = 0

for sourceImgName in list(res.keys()) : 
    res[sourceImgName] = sorted(res[sourceImgName], key=lambda s: s[1], reverse=True)
    predTop10 = []
    for i in range(len(res[sourceImgName])) : 
        if len(predTop10) < 10 and label['annotation'][res[sourceImgName][i][0]] not in predTop10 : 
            predTop10.append(label['annotation'][res[sourceImgName][i][0]])
    if label['annotation'][sourceImgName] == predTop10[0] : 
        truePosCount += 1
    if label['annotation'][sourceImgName] in predTop10 : 
        truePosTop10Count += 1
        
res['accuracy'] = truePosCount / float(nbSourceImg)
msg = '***** Final accuracy is {:.3f}, Top 10 {:.3f}*****'.format(res['accuracy'], truePosTop10Count / nbSourceImg)
print (msg)


sourceImgNameList = sorted(list(res.keys()))
if args.outDir : 
    os.mkdir(args.outDir)
    
    
    for j, sourceImgName in enumerate(sourceImgNameList) :
        if sourceImgName == 'accuracy' :
            continue  
        tmpOutDir = os.path.join(args.outDir, str(j)) 
        os.mkdir(tmpOutDir)
        src = os.path.join(args.queryDir, sourceImgName)
        dst = os.path.join(tmpOutDir, 'query.jpg')
        copyfile(src, dst)
        top5 = []
        for i in range(len(res[sourceImgName])) :
            if len(top5) < 5 and label['annotation'][res[sourceImgName][i][0]] not in top5 : 
                top5.append(label['annotation'][res[sourceImgName][i][0]])
                out = 'Rank{:d}_{:d}.jpg'.format(i, 1) if label['annotation'][sourceImgName] == label['annotation'][res[sourceImgName][i][0]] else 'Rank{:d}_{:d}.jpg'.format(i, 0)
                out = os.path.join( tmpOutDir, out)
                out2 = os.path.join( tmpOutDir, 'Org{:d}.jpg'.format(i) )
                outils.drawPair(os.path.join(args.searchDir, res[sourceImgName][i][0]), out, out2, res[sourceImgName][i][3], res[sourceImgName][i][2])
        resSave = {}
        for sourceImgName in res :
            if sourceImgName != 'accuracy': 
                resSave[sourceImgName] = [[res[sourceImgName][i][1], res[sourceImgName][i][0], res[sourceImgName][i][2], 0] for i in range(len(res[sourceImgName]))]
            
        with open(os.path.join(args.outDir, 'res.json'), 'w') as f : 
            json.dump(resSave, f)        
         
    






