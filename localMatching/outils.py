import numpy as np
from skimage import measure
from torch.autograd import Variable
from scipy.signal import convolve2d
import torch
import PIL.Image as Image 
import cv2
import os 
## resize the image to the indicated scale
def ResizeImg(featMax, featMin, minNet, strideNet, w, h) :

    ratio = float(w)/h
    if ratio < 1 : 
        featH = featMax 
        featW = max(round(ratio * featH), featMin )
        
    else : 
        featW = featMax 
        featH = max(round(featW/ratio), featMin )
    resizeW = (featW - 1) * strideNet + minNet
    resizeH = (featH - 1) * strideNet + minNet

    return int(resizeW), int(resizeH), float(resizeW)/w, float(resizeH)/h
    
    

    
def FeatImgRef(I, scaleImgRef, minNet, strideNet, margin, transform, model, featChannel, eta = 1e-7) : 

    # Resize image
    pilImgW, pilImgH = I.size
    resizeW, resizeH, wRatio, hRatio =  ResizeImg(scaleImgRef, 2 * margin + 1, minNet, strideNet, pilImgW, pilImgH)
    pilImg = I.resize((resizeW, resizeH))
    
    ## Image feature
    feat=transform(pilImg).unsqueeze(0)
    feat = feat.cuda()
    with torch.no_grad() : 
        feat = model.forward(feat).data
    featW, featH = feat.size()[2], feat.size()[3] ## attention : featW and featH correspond to pilImgH and pilImgW respectively 
    feat = feat / (eta + (torch.sum(feat **2, dim = 1, keepdim=True).expand(feat.size())**0.5) )
    feat = feat[:, :, margin : featW - margin, margin : featH - margin].contiguous().view(featChannel, -1)
    
    feat = feat.view(featChannel, -1)
    
    ## Other information
    bbox = [margin * strideNet / wRatio, margin * strideNet / hRatio, (featH - margin) * strideNet / wRatio, (featW - margin) * strideNet / hRatio]
    imgBbox = map(int, bbox)
    featW, featH = featW - 2 * margin, featH - 2 * margin
    listW = (torch.range(0, featW -1, 1)).unsqueeze(1).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
    listH = (torch.range(0, featH -1, 1)).unsqueeze(0).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
    
    return feat, pilImgW, pilImgH, featW, featH, listW, listH, imgBbox
    
def imgFeat(minNet, strideNet, I, model, transform, scale, eta) : 
    w,h = I.size
    new_w, new_h, _, _ = ResizeImg(scale, 1, minNet, strideNet, w, h)
    Ifeat = transform(I.resize((new_w, new_h))).unsqueeze(0).cuda()
    with torch.no_grad() : 
        Ifeat = model(Ifeat).data
        Ifeat = Ifeat.data
        Ifeat_norm = torch.sum(Ifeat ** 2, dim = 1, keepdim=True) ** 0.5 + eta
        Ifeat_I2 = Ifeat / Ifeat_norm.expand(Ifeat.size())
    
    return Ifeat_I2
    
def MatchPair(minNet, strideNet, model, transform, scales, feat1, feat1W, feat1H, img2Dir, img2Name, listW, listH, featChannel, tolerance, margin, eta = 1e-7, RefFeat={}, flip = False):

    match1 = []
    match2 = []
    similarity = []
    gridSize = []
    nbFeat = feat1W * feat1H
    bestScore = 0
    if len(RefFeat) == 0 : 
        I2Path = os.path.join(img2Dir, img2Name)
        I2 = Image.open(I2Path).convert('RGB')
        if flip : 
            I2 = I2.transpose(Image.FLIP_LEFT_RIGHT)
    for i in range(len(scales)) : 
        # Normalized I2 feature
        if len(RefFeat) > 0 : 
            tmp_I2 = RefFeat[img2Name][i]
        else :  
            tmp_I2 = imgFeat(minNet, strideNet, I2, model, transform, scales[i], eta)
        
        # Hough Transformation Grid, only for the current scale
        tmp_feat_w, tmp_feat_h = tmp_I2.shape[2], tmp_I2.shape[3]
        tmp_transformation = np.zeros((feat1W + tmp_feat_w, feat1H + tmp_feat_h))
        
        # I2 spatial information
        tmp_nbFeat = tmp_feat_w * tmp_feat_h
        tmp_I2 = tmp_I2.view(featChannel, -1).contiguous().transpose(0, 1)
        tmp_w = (torch.range(0, tmp_feat_w-1,1)).unsqueeze(1).expand(tmp_feat_w, tmp_feat_h).contiguous().view(-1).type(torch.LongTensor)
        tmp_h = (torch.range(0, tmp_feat_h-1, 1)).unsqueeze(0).expand(tmp_feat_w, tmp_feat_h).contiguous().view(-1).type(torch.LongTensor)

        # Feature Similarity
        score = torch.mm(tmp_I2, feat1)
        
        # Top1 match for both images
        topk0_score, topk0_index = score.topk(k=1, dim = 0)
        topk1_score, topk1_index = score.topk(k=1, dim = 1)
        
        index0 = torch.cuda.FloatTensor(tmp_nbFeat, nbFeat).fill_(0).scatter_(0, topk0_index, topk0_score)
        index1 = torch.cuda.FloatTensor(tmp_nbFeat, nbFeat).fill_(0).scatter_(1, topk1_index, topk1_score)
        intersectionScore = index0 * index1
        intersection = intersectionScore.nonzero()
        
        for item in intersection : 
            i2, i1 = item[0], item[1]
            w1, h1, w2, h2 = float(listW[i1].item()), float(listH[i1].item()), float(tmp_w[i2].item()), float(tmp_h[i2].item())
            
            # Store all the top1 matches
            match1.append([(w1 + 0.49 + margin) / (feat1W + 2 * margin), (h1 + 0.49 + margin) / (feat1H + 2 * margin)])
            match2.append([(w2 + 0.49) / tmp_feat_w, (h2 + 0.49) / tmp_feat_h])
            gridSize.append([1. / tmp_feat_w, 1. / tmp_feat_h])
            
            similarity.append(intersectionScore[i2, i1].item() ** 0.5)
        
    
    match1, match2, similarity, gridSize = np.array(match1), np.array(match2), np.array(similarity), np.array(gridSize)
    
    return np.hstack((match1, np.ones((match1.shape[0], 1)))), np.hstack((match2, np.ones((match2.shape[0], 1)))), similarity, gridSize

def Affine(X, Y):
	nb_points = X.shape[0]
	H21 = np.linalg.lstsq(Y, X[:, :2])[0]
	H21 = H21.T
	H21 = np.array([[H21[0, 0], H21[0, 1], H21[0, 2]], 
					[H21[1, 0], H21[1, 1], H21[1, 2]],
					[0, 0, 1]])
	return H21

def Hough(X, Y) : 
    nb_points = X.shape[0]
    ones = np.ones((nb_points, 1))
    H21x = np.linalg.lstsq(np.hstack((Y[:, 0].reshape((-1, 1)), ones)), X[:, 0])[0]
    H21y = np.linalg.lstsq(np.hstack((Y[:, 1].reshape((-1, 1)), ones)), X[:, 1])[0]
    
    H21 = np.array([[H21x[0], 0, H21x[1]], 
                    [0, H21y[0], H21y[1]],
                    [0, 0, 1]])
    return H21

def Prediction(X, Y, H21) : 

    estimX = (np.dot(H21, Y.T)).T
    estimX = estimX / estimX[:, 2].reshape((-1, 1))
    return np.sum((X[:, :2]- estimX[:, :2])**2, axis=1)**0.5
    	    
    
def ScorePosIdentity(match1, match2, score, gridSize, tolerance) : 
    
    #All The Data
    
    error = np.sum((match2 - match1) ** 2, axis = 1)**0.5
    if len(match1) < 4 : 
        return 0, {}
    scoreTotal = score * np.exp(-1 * error ** 2 / tolerance ** 2)
    keys = [(match1[i][0], match1[i][1]) for i in range(len(match1))]
    bestInlier = {k : [0, 0, 0] for k in keys}
    for j in range(len(keys)) : 
        if scoreTotal[j] > bestInlier[keys[j]][1] : 
            bestInlier[keys[j]] = [match2[j], scoreTotal[j], gridSize[j]]
            
    return sum([item[1] for item in bestInlier.values()]), bestInlier
    
def ScorePosHough(match1, match2, score, gridSize, tolerance, nbIter = 1000) : 
    
    #All The Data
    bestScore, bestInlier, bestH = 0, {}, []
    matchCombine = np.arange(len(match1))
    if len(matchCombine) < 10 : 
        return bestScore, bestInlier
    keys = [(match1[i][0], match1[i][1]) for i in range(len(match1))]
    
    for i in range(nbIter) : 
        index = np.random.choice(matchCombine, 2, replace=False)
        H21 = Hough(match1[index], match2[index])
        error = Prediction(match1, match2, H21)
        scoreTotal = score * np.exp(-1 * error ** 2 / tolerance ** 2)
        inlier = {k : 0 for k in keys}
        for j in range(len(keys)) : 
            inlier[keys[j]] = max(inlier[keys[j]], scoreTotal[j])
        scoreTotal = sum(inlier.values())
        if scoreTotal > bestScore : 
            bestScore, bestH = scoreTotal, H21
            
    if len(bestH) > 0 : 
        error = Prediction(match1, match2, bestH)
        scoreTotal = score * np.exp(-1 * error ** 2 / tolerance ** 2)
        bestInlier = {k : [0, 0, 0] for k in keys}
        for j in range(len(keys)) : 
            if scoreTotal[j] > bestInlier[keys[j]][1] : 
                bestInlier[keys[j]] = [match2[j], scoreTotal[j], gridSize[j]]
        
            
    return bestScore, bestInlier
    
def ScorePosAffine(match1, match2, score, gridSize, tolerance, nbIter = 1000) : 
    
    #All The Data
    bestScore, bestInlier, bestH = 0, {}, []
    matchCombine = np.arange(len(match1))
    if len(matchCombine) < 10 : 
        return bestScore, bestInlier
    keys = [(match1[i][0], match1[i][1]) for i in range(len(match1))]
    
    for i in range(nbIter) : 
        index = np.random.choice(matchCombine, 3, replace=False)
        H21 = Affine(match1[index], match2[index])
        error = Prediction(match1, match2, H21)
        scoreTotal = score * np.exp(-1 * error ** 2 / tolerance ** 2)
        inlier = {k : 0 for k in keys}
        for j in range(len(keys)) : 
            inlier[keys[j]] = max(inlier[keys[j]], scoreTotal[j])
        scoreTotal = sum(inlier.values())
        if scoreTotal > bestScore : 
            bestScore, bestH = scoreTotal, H21
            
    if len(bestH) > 0 : 
        error = Prediction(match1, match2, bestH)
        scoreTotal = score * np.exp(-1 * error ** 2 / tolerance ** 2)
        bestInlier = {k : [0, 0, 0] for k in keys}
        for j in range(len(keys)) : 
            if scoreTotal[j] > bestInlier[keys[j]][1] : 
                bestInlier[keys[j]] = [match2[j], scoreTotal[j], gridSize[j]]
        
            
    return bestScore, bestInlier
    

    


## Blur the mask
def BlurMask(mask) : 

    mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
    mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
    mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
    
    return mask
    
def drawPair(imgPath2, out2, outOrg, flipHorizontal, inlier) :  

    I2 = Image.open(imgPath2).convert('RGB')
    
    if flipHorizontal: 
        I2 = I2.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    w2, h2 = I2.size
    I2RGB = cv2.cvtColor(np.array(I2), cv2.COLOR_RGB2BGR)
    M2 = np.zeros((h2, w2))
    if len(inlier.keys()) == 0 : 
        cv2.imwrite(out2, I2RGB)
        return 
        
    for pair1 in inlier : 
        pair2, score, gridSize2 = inlier[pair1]
        
        M2[int((pair2[0] - gridSize2[0] / 2) * h2) : int((pair2[0] + gridSize2[0] / 2) * h2) + 1, int((pair2[1] - gridSize2[1] / 2) * w2) : int((pair2[1] + gridSize2[1] / 2) * w2) + 1] = score
    
    M2 = (M2 - M2.min()) / (M2.max() - M2.min())
    
    M2 = cv2.applyColorMap((M2 * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    cv2.imwrite(out2, cv2.addWeighted(M2, 0.5, I2RGB, 0.5, 0))
    cv2.imwrite(outOrg, I2RGB)
    

