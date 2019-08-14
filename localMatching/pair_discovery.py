# coding: utf-8
import sys
import torch 
import cv2
import outils
import os
from torchvision import datasets, transforms,models
import numpy as np 
from tqdm import tqdm

import argparse
import warnings
import PIL.Image as Image
from scipy.misc import imresize


from scipy.signal import convolve2d
import time

if not sys.warnoptions:
	warnings.simplefilter("ignore")



def PairDiscovery(img1Name, img1Dir, img2Name, img2Dir, model, transform, tolerance, margin, scaleImgRef, scaleList, eta=1e-7, featLayer = 'conv4', scoreType = 'ALL', RefFeat = {}, flip = False) : 
	
	if scoreType == 'Identity' : 
		ScorePos = outils.ScorePosIdentity 
	elif scoreType == 'Hough' : 
		ScorePos = outils.ScorePosHough 
	elif scoreType == 'Affine' : 
		ScorePos = outils.ScorePosAffine 
		
	
	strideNet = 16
	minNet = 15
	
	featChannel = 256 if featLayer == 'conv4' else 512
	
	img1Path = os.path.join(img1Dir, img1Name)
	
	I1 = Image.open(img1Path).convert('RGB')
		
	
	feat1, pilImg1W, pilImg1H, feat1W, feat1H, list1W, list1H, img1Bbox  = outils.FeatImgRef(I1, scaleImgRef, minNet, strideNet, margin, transform, model, featChannel, eta)
	toleranceRef = tolerance / scaleImgRef
	
	
	match1, match2, similarity, gridSize = outils.MatchPair(minNet, strideNet, model, transform, scaleList, feat1, feat1W, feat1H, img2Dir, img2Name, list1W, list1H, featChannel, tolerance, margin, eta, RefFeat, flip)
	
	if len(match2) < 3  : 
		return 0., {}
	
	bestScore, inlier = ScorePos(match1, match2, similarity, gridSize, toleranceRef)
	
	return bestScore, inlier

