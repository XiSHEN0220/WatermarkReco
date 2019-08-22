import numpy as np
import PIL.Image as Image
import torch
import os
from itertools import product, combinations
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm



## Given a featMax (maximum dimensions in the feature map)
## The function calculate the output size to resize the image with keeping the aspect reatio
## The minimum dimensions in the feature map is at least featMin

def ResizeImg(featMin, featMax, minNet, strideNet, w, h) :

    ratio = float(w)/h
    if ratio < 1 :
        featH = featMax
        featW = max(round(ratio * featH), featMin)
    else :
        featW = featMax
        featH = max(round(featW/ratio), featMin)

    resizeW, resizeH = int((featW -1) * strideNet + minNet), int((featH -1) * strideNet + minNet)

    return resizeW, resizeH

class InfiniteSampler():
    def __init__(self, img_list):
        self.img_list = img_list
    def loop(self):

        while True:
            for i, data in enumerate(self.img_list) :
                yield i, data
            self.img_list = np.random.permutation(self.img_list)

## Random Query Feature

def RandomQueryFeat(nbPatchTotal, featChannel, searchRegion, imgFeatMin, minNet, strideNet, transform, net, searchDir, margin, imgList, useGpu, queryScale, isTrain=True) :

    featQuery = torch.cuda.FloatTensor(nbPatchTotal, featChannel, searchRegion, searchRegion) # Store feature
    img_sampler = InfiniteSampler(imgList)
    count = 0
    queryNameList = []
    
    for (i, img_name) in tqdm(img_sampler.loop()) :
        if count == nbPatchTotal :
            break

        ## resize image
        I = Image.open(os.path.join(searchDir, img_name)).convert('RGB')
        w,h = I.size
        queryNameList.append(img_name)
        
        new_w, new_h = ResizeImg(imgFeatMin, queryScale, minNet, strideNet, w, h)
        I = I.resize((new_w, new_h))

        ## Image Feature
        I_data = transform(I).unsqueeze(0)
        I_data = I_data.cuda() if useGpu else I_data ## set cuda datatype?
        
        with torch.no_grad() : 
            I_data = net(I_data).data ## volatile, since do not need gradient

        ## Query feature + Query Information
        feat_w, feat_h = I_data.shape[2], I_data.shape[3]
        feat_w_pos, feat_h_pos = np.arange(margin, feat_w - margin - searchRegion, 1), np.arange(margin, feat_h - margin - searchRegion, 1)
        feat_w_pos = np.random.choice(feat_w_pos, 1)[0] if isTrain else feat_w_pos[i % len(feat_w_pos)]
        feat_h_pos = np.random.choice(feat_h_pos , 1)[0] if isTrain else feat_h_pos[i % len(feat_h_pos)]
        featQuery[count] = I_data[:, :, feat_w_pos : feat_w_pos + searchRegion, feat_h_pos : feat_h_pos + searchRegion].clone()

        count += 1

    return Variable(featQuery), queryNameList

## Cosine similarity Implemented as a Convolutional Layer
## Note: we don't normalize kernel
def CosineSimilarity(img_feat, kernel, kernel_one) :

    dot = F.conv2d(img_feat, kernel, stride = 1)
    img_feat_norm = F.conv2d(img_feat ** 2, kernel_one, stride = 1) ** 0.5 + 1e-7
    score = dot/img_feat_norm.expand(dot.size())

    return score.data

## Cosine similarity and we only keep topK score
def CosineSimilarityTopK(img_feat, img_feat_norm, kernel, K) :

    dot = F.conv2d(img_feat, kernel, stride = 1)
    score = dot/img_feat_norm.expand_as(dot)
    _, _, score_w, score_h =  score.size()
    score = score.view(kernel.size()[0], score_w * score_h)
    topk_score, topk_index = score.topk(k = K, dim = 1)
    topk_w, topk_h = topk_index / score_h, topk_index % score_h

    return topk_score, topk_w, topk_h

## Feature Normalization, Divided by L2 Norm
def Normalization(feat) :

    feat_norm = (torch.sum(torch.sum(torch.sum(feat ** 2, dim = 1), dim = 1), dim = 1) + 1e-7) ** 0.5
    feat_normalized = feat / feat_norm.view(feat.size()[0], 1, 1, 1).expand_as(feat)

    return feat_normalized

## Image feature for searching image :
##                                     1. features in different scales;
##                                     2. remove feature in the border
def SearchImgFeat(searchDir, margin, searchRegion, scales, minNet, strideNet, transform, model, searchName, useGpu) :

    searchFeat = {}
    I = Image.open(os.path.join(searchDir, searchName)).convert('RGB')
    w,h = I.size

    for s in scales :

        new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, s, minNet, strideNet, w, h)
        I_pil = I.resize((new_w, new_h))
        I_data = transform(I_pil).unsqueeze(0)
        with torch.no_grad() : 
            I_data = I_data.cuda() if useGpu else I_data
        feat = model(I_data).data
        feat_w, feat_h = feat.shape[2], feat.shape[3]
        searchFeat[s] = feat[:, :, margin : feat_w - margin, margin : feat_h - margin].clone()

    return searchFeat

def RetrievalRes(nbPatchTotal, imgList, searchDir, margin, searchRegion, scales, minNet, strideNet, transform, net, featQuery, useGpu, K=20) :

    resScale = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList))) # scale
    resW = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList)))     # feat_w
    resH = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList)))     # feat_h
    resScore = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList))) # score

    variableAllOne =  Variable(torch.ones(1, featQuery.size()[1], featQuery.size()[2], featQuery.size()[3])).cuda()

    for i in tqdm(range(len(imgList))) :

        search_name = imgList[i]
        searchFeat = SearchImgFeat(searchDir, margin, searchRegion, scales, minNet, strideNet, transform, net, search_name, useGpu)

        tmpScore = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
        tmpH = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
        tmpW = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
        tmpScale = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))

        for j, scale in enumerate(list(searchFeat.keys())) :

            featImg = searchFeat[scale]
            score = CosineSimilarity(featImg, featQuery, variableAllOne)

            # Update tmp matrix
            outW = score.size()[2]
            outH = score.size()[3]
            score = score.view(score.size()[1], outW * outH)
            score, index= score.max(1)
            tmpW[:, j] = index/outH
            tmpH[:, j] = index%outH
            tmpScore[:, j] = score
            tmpScale[:, j] = float(scale)

        tmpScore, tmpScaleIndex = tmpScore.max(1)
        tmpScaleIndex = tmpScaleIndex.unsqueeze(1)

        # Update res matrix, only keep top 10
        resScore[:, i] = tmpScore
        resScale[:, i] = torch.gather(tmpScale, 1, tmpScaleIndex).squeeze(1)
        resW[:, i] = torch.gather(tmpW, 1, tmpScaleIndex).squeeze(1)
        resH[:, i] = torch.gather(tmpH, 1, tmpScaleIndex).squeeze(1)

    # Get Topk Matrix
    topkValue, topkImg = resScore.topk(k = K, dim = 1) ## Take Top10 pairs
    topkScale = torch.gather(resScale, dim = 1, index = topkImg)
    topkW = torch.gather(resW, 1, topkImg)
    topkH = torch.gather(resH, 1, topkImg)

    topkW = topkW.type(torch.cuda.LongTensor) if useGpu else topkW.type(torch.LongTensor)
    topkH = topkH.type(torch.cuda.LongTensor) if useGpu else topkH.type(torch.LongTensor)

    return topkImg, topkScale, topkValue, topkW + margin, topkH + margin





def TrainPair(searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, useGpu, featChannel, searchRegion, minNet, strideNet, labelCategory, K, queryNameList, tolerance, isTrain = True) :

    posPair = []
    negPair = []
    for i in tqdm(range(topkImg.size()[0])) :
        posPairIndex = []
        tmpPosPair = []
        tmpNegaPair = []
        
        for j in range(K) : 
            if labelCategory[imgList[topkImg[i, j]]] == labelCategory[queryNameList[i]] : 
                posPairIndex.append(j)
        
        if len(posPairIndex) < 2 : 
            continue
        
        posPairIndex = [[posPairIndex[j], posPairIndex[k]] for j, k in combinations(range(len(posPairIndex)), 2)]
        for j in range(len(posPairIndex)) : 
            Posw1, Posh1, s1, name1 = topkW[i, posPairIndex[j][0]].item(), topkH[i, posPairIndex[j][0]].item(), topkScale[i, posPairIndex[j][0]].item(), topkImg[i, posPairIndex[j][0]].item()
            Posw2, Posh2, s2, name2 = topkW[i, posPairIndex[j][1]].item(), topkH[i, posPairIndex[j][1]].item(), topkScale[i, posPairIndex[j][1]].item(), topkImg[i, posPairIndex[j][1]].item()
            
            ## feature dimension of I1 and I2
            I1 = Image.open(os.path.join(searchDir, imgList[name1])).convert('RGB')
            w,h = I1.size
            new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, s1, minNet, strideNet, w, h)
            Fw1, Fh1 = (new_h - minNet) / strideNet + 1, (new_w - minNet) / strideNet + 1
            
            I2 = Image.open(os.path.join(searchDir, imgList[name2])).convert('RGB')
            w,h = I2.size
            new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, s2, minNet, strideNet, w, h)
            Fw2, Fh2 = (new_h - minNet) / strideNet + 1, (new_w - minNet) / strideNet + 1
            
            error = ((Posw1 / float(Fw1) - Posw2 / float(Fw2)) ** 2 + (Posh1 / float(Fh1) - Posh2 / float(Fh2)) ** 2) ** 0.5 
            if error <= tolerance : 
                tmpPosPair.append([i, posPairIndex[j][0], posPairIndex[j][1]])
        
                
        
        if len(tmpPosPair) > 0:
            negPairIndex = []
            for j,k in tqdm(combinations(range(K), 2)) : 
                if [j,k] not in posPairIndex and len(negPairIndex) <  len(tmpPosPair) :
                    negPairIndex.append([j, k])
                if len(negPairIndex) ==  len(tmpPosPair) : 
                    break
                    
            for j in range(len(negPairIndex)) : 
                tmpNegaPair.append([i, negPairIndex[j][0], negPairIndex[j][1]])
        if isTrain : 
            nbPair = min(len(tmpNegaPair), len(tmpPosPair))
            if nbPair > 0 : 
                posPair = posPair + tmpPosPair[- nbPair : ]
                negPair = negPair + tmpNegaPair[: nbPair]
        else : 
            posPair = posPair + tmpPosPair
            
    return np.array(posPair).astype(np.int64), np.array(negPair).astype(np.int64)

def cropPatch(searchDir, imgName, scale, featW, featH, margin, minNet, strideNet, searchRegion, saveName) : 
    
    I = Image.open(os.path.join(searchDir, imgName)).convert('RGB')
    w,h = I.size
    new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, scale, minNet, strideNet, w, h)
    patchSize = 2 * margin * strideNet + minNet
    I = I.resize((new_w, new_h))
    left = (featH - margin) * strideNet
    top = (featW - margin) * strideNet
    right = left + patchSize
    bottom = top + patchSize
    I = I.crop([left, top, right, bottom])
    return I.save(saveName, quality=np.random.randint(50, 100))
    
def saveTrainImgPair(posInfo, negInfo, tmpTrainDir, margin, searchDir, imgList, topkImg, topkScale, topkW, topkH, minNet, strideNet, searchRegion) :
    
    tmpPos = os.path.join(tmpTrainDir, 'posPair')
    if os.path.exists(tmpPos) : 
        cmd = 'rm -r {}'.format(tmpPos)
        os.system(cmd)
    os.mkdir(tmpPos)
    
    tmpNeg = os.path.join(tmpTrainDir, 'negPair')
    if os.path.exists(tmpNeg) : 
        cmd = 'rm -r {}'.format(tmpNeg)
        os.system(cmd)
    os.mkdir(tmpNeg)
    
    msg = 'Generating Training Patches...'
    print (msg)
    for i in range(len(posInfo)) : 
        cropPatch(searchDir, imgList[topkImg[int(posInfo[i, 0]), int(posInfo[i, 1])].item()], topkScale[int(posInfo[i, 0]), int(posInfo[i, 1])].item(), topkW[int(posInfo[i, 0]), int(posInfo[i, 1])].item(), topkH[int(posInfo[i, 0]), int(posInfo[i, 1])].item(), margin, minNet, strideNet, searchRegion, os.path.join(tmpPos, '{:d}_1.jpg'.format(i)))
        
        cropPatch(searchDir, imgList[topkImg[int(posInfo[i, 0]), int(posInfo[i, 2])].item()], topkScale[int(posInfo[i, 0]), int(posInfo[i, 2])].item(), topkW[int(posInfo[i, 0]), int(posInfo[i, 2])].item(), topkH[int(posInfo[i, 0]), int(posInfo[i, 2])].item(), margin, minNet, strideNet, searchRegion, os.path.join(tmpPos, '{:d}_2.jpg'.format(i)))
        
        cropPatch(searchDir, imgList[topkImg[int(negInfo[i, 0]), int(negInfo[i, 1])].item()], topkScale[int(negInfo[i, 0]), int(negInfo[i, 1])].item(), topkW[int(negInfo[i, 0]), int(negInfo[i, 1])].item(), topkH[int(negInfo[i, 0]), int(negInfo[i, 1])].item(), margin, minNet, strideNet, searchRegion, os.path.join(tmpNeg, '{:d}_1.jpg'.format(i)))
        
        cropPatch(searchDir, imgList[topkImg[int(negInfo[i, 0]), int(negInfo[i, 2])].item()], topkScale[int(negInfo[i, 0]), int(negInfo[i, 2])].item(), topkW[int(negInfo[i, 0]), int(negInfo[i, 2])].item(), topkH[int(negInfo[i, 0]), int(negInfo[i, 2])].item(), margin, minNet, strideNet, searchRegion, os.path.join(tmpNeg, '{:d}_2.jpg'.format(i)))
        
        

## Process training pairs, sampleIndex dimension: iterEpoch * batchSize
def DataShuffle(sample, batchSize) :

    nbSample = len(sample)
    iterEpoch = nbSample / batchSize

    permutationIndex = np.random.permutation(range(nbSample))
    sampleIndex = permutationIndex.reshape(( iterEpoch, batchSize)).astype(int)

    return sampleIndex

## Positive loss for a pair of positive matching
def PosCosineSimilaritytop1(feat1, feat2, pos_w1, pos_h1, pos_w2, pos_h2, variableAllOne) :

    feat1x1 = feat1[:, :, pos_w1, pos_h1].clone().contiguous()
    feat1x1 = feat1x1 / ((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)

    tmp_pos_w2 = max(pos_w2 - 1, 0)
    tmp_pos_h2 = max(pos_h2 - 1, 0)
    tmp_end_w2 = min(pos_w2 + 2, feat2.size()[2])
    tmp_end_h2 = min(pos_h2 + 2, feat2.size()[3])

    featRec = feat2[:, :, tmp_pos_w2 : tmp_end_w2, tmp_pos_h2 : tmp_end_h2].clone().contiguous()
    featRecNorm = F.conv2d(featRec ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7

    return CosineSimilarityTopK(featRec, featRecNorm, feat1x1.unsqueeze(2).unsqueeze(3), K = 1)[0][0]



## Negative sample loss for a pair of negative matching
def NegaCosineSimilaritytopk(feat1, feat2, norm2, pos_w1, pos_h1, variableAllOne, topKLoss) :

    feat1x1 = feat1[:, :, pos_w1, pos_h1].clone().contiguous()
    feat1x1 = feat1x1 / ((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)
    negaTopKLoss = CosineSimilarityTopK(feat2, norm2, feat1x1.unsqueeze(2).unsqueeze(3), K = topKLoss)[0]

    return torch.mean(negaTopKLoss)




def PairPos(pos_w1, pos_h1, pos_w2, pos_h2, trainRegion) :


    pos1 = [(pos_w1 , pos_h1 ), (pos_w1, pos_h1 + trainRegion - 1), (pos_w1 + trainRegion - 1, pos_h1), (pos_w1 + trainRegion - 1, pos_h1 + trainRegion - 1)]
    pos2 = [(pos_w2 , pos_h2 ), (pos_w2, pos_h2 + trainRegion - 1), (pos_w2 + trainRegion - 1, pos_h2), (pos_w2 + trainRegion - 1, pos_h2 + trainRegion - 1)]

    return pos1, pos2

def PosNegaSimilarity(posPair, posIndex, topkImg, topkScale, topkW, topkH, searchDir, imgList, minNet, strideNet, net, transform, searchRegion, trainRegion, margin, featChannel, useGpu, topKLoss) :

    # Pair information: image name, scale, W, H
    pair = posPair[posIndex]
    queryIndex = int(pair[0])
    pairIndex = [int(pair[1]), int(pair[2])]
    info1 = (topkImg[queryIndex, pairIndex[0]], topkScale[queryIndex, pairIndex[0]], topkW[queryIndex, pairIndex[0]] - (trainRegion + 1) / 2 + 1, topkH[queryIndex, pairIndex[0]] - (trainRegion + 1) / 2 + 1)
    info2 = (topkImg[queryIndex, pairIndex[1]], topkScale[queryIndex, pairIndex[1]], topkW[queryIndex, pairIndex[1]] - (trainRegion + 1) / 2 + 1 , topkH[queryIndex, pairIndex[1]] - (trainRegion + 1) / 2 + 1)

    ## features of pair images
    I1 = Image.open(os.path.join(searchDir, imgList[info1[0]])).convert('RGB')
    w,h = I1.size
    new_w, new_h = ResizeImg(margin * 2 + searchRegion + 1, info1[1], minNet, strideNet, w, h)
    feat1 = net(transform(I1.resize((new_w, new_h))).unsqueeze(0).cuda()) if useGpu else net(transform(I1.resize((new_w, new_h))).unsqueeze(0))

    I2 = Image.open(os.path.join(searchDir, imgList[info2[0]])).convert('RGB')
    w,h = I2.size
    new_w, new_h = ResizeImg(margin * 2 + searchRegion + 1, info2[1], minNet, strideNet, w, h)
    feat2 = net(transform(I2.resize((new_w, new_h)).unsqueeze(0).cuda())) if useGpu else net(transform(I2.resize((new_w, new_h))).unsqueeze(0))



    variableAllOne = torch.ones(1, featChannel, 1, 1).cuda() if useGpu else  torch.ones(1, featChannel, 1, 1)

    norm2 = F.conv2d(feat2 ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7
    norm1 = F.conv2d(feat1 ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7

    pos1, pos2 = PairPos(info1[2], info1[3], info2[2], info2[3], trainRegion)

    posTop1Similarity = []
    negaTopKSimilarity = []

    for (pair1, pair2) in zip(pos1, pos2) :

        posTop1Similarity.append( PosCosineSimilaritytop1(feat1, feat2, pair1[0], pair1[1], pair2[0], pair2[1], variableAllOne) )
        nega1 = NegaCosineSimilaritytopk(feat1, feat2, norm2, pair1[0], pair1[1], variableAllOne, topKLoss)
        nega2 = NegaCosineSimilaritytopk(feat2, feat1, norm1, pair2[0], pair2[1], variableAllOne, topKLoss)
        if nega1.data[0] > nega2.data[0] :
            negaTopKSimilarity.append(nega1)
        else :
            negaTopKSimilarity.append(nega2)


    return posTop1Similarity, negaTopKSimilarity
    
def CosSimilarity(p1, p2, n1, n2, margin, eta=1e-1) : 
    p1 = p1[:, :, margin, margin].clone()
    p2 = p2[:, :, margin, margin].clone()
    n1 = n1[:, :, margin, margin].clone()
    n2 = n2[:, :, margin, margin].clone()
    p1 = p1 / (torch.sum(p1 ** 2, dim = 1, keepdim=True).expand(p1.size())**0.5 + eta)
    p2 = p2 / (torch.sum(p2 ** 2, dim = 1, keepdim=True).expand(p2.size())**0.5 + eta)
    n1 = n1 / (torch.sum(n1 ** 2, dim = 1, keepdim=True).expand(n1.size())**0.5 + eta)
    n2 = n2 / (torch.sum(n2 ** 2, dim = 1, keepdim=True).expand(n2.size())**0.5 + eta)
    return torch.sum(p1 * p2, dim = 1), torch.sum(n1 * n2, dim = 1)
    
