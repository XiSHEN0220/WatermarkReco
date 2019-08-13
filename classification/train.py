
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR

import ujson
import numpy as np
import os
import argparse

from outils import progress_bar
from dataloader import TrainLoader, ValLoader


class FC(nn.Module):
	def __init__(self, dropoutRatio, inFeat, nbCls):
		super(FC, self).__init__()
		self.dropout = nn.Dropout(dropoutRatio)
		self.fc1 = nn.Linear(inFeat, nbCls)
		
	def forward(self, x):

		x = self.dropout(x)
		x = self.fc1(x)
		
		return x
				
		
parser = argparse.ArgumentParser(description='PyTorch WaterMark Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--outDir', type=str, help='output directory')
parser.add_argument('--trainDir', type = str, default = '../data/watermark/A_classification/train/', help='train image directory')
parser.add_argument('--valDir', type = str, default = '../data/watermark/A_classification/test/', help='val image directory')
parser.add_argument('--batchSize', type = int, default = 64, help='batch size')
parser.add_argument('--nbEpoch', type = int, default = 300, help='np epoch')
parser.add_argument('--dropoutRatio', type = float, default = 0.7, help='dropout ratio')
parser.add_argument('--nbCls', type = int, default=100, help='nb of output classes')
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--cj', type = float, default=0.4, help='color jitter parameter')
parser.add_argument('--resumePth', type = str, help='resumePth')





args = parser.parse_args()
print (args)
bestAcc = 0  # best test accuracy

normalize = transforms.Normalize(mean=[ 0.75,0.70,0.65],std=[ 0.14,0.15,0.16]) 

trainTransform = transforms.Compose([
									transforms.RandomResizedCrop(224),
									transforms.ColorJitter(brightness=args.cj, contrast=args.cj, saturation=args.cj, hue=args.cj/2),
									transforms.RandomHorizontalFlip(),
									transforms.RandomVerticalFlip(),
									transforms.ToTensor(),
									normalize,
								])
								
		
valTransform = transforms.Compose([
									transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize,
								])
		
trainLoader = TrainLoader(args.batchSize, args.trainDir, trainTransform)
valLoader = ValLoader(args.batchSize, args.valDir, valTransform)


randomSeed = 123
np.random.seed(randomSeed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(randomSeed)

net = models.resnet18()
if args.resumePth : 
	net.fc = FC(args.dropoutRatio, 512, 100)
	net.load_state_dict(torch.load(args.resumePth))
	msg = 'Loading weight from {}'.format(args.resumePth)
	print (msg)
	net.fc = FC(args.dropoutRatio, 512, args.nbCls)
	
	
if args.cuda : 
	net.cuda()

if not os.path.isdir(args.outDir):
	os.mkdir(args.outDir)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.fc.parameters(), lr=args.lr, betas=(0.5, 0.999)) if args.resumePth  else torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))


testAccLog = []
trainAccLog = []

# Training
def train(epoch, useGpu): 
	msg = '\nEpoch: {:d}'.format(epoch)
	print (msg)
	net.train()
	trainLoss = 0
	correct = 0
	total = 0
		
	for batchIdx, (inputs, targets) in enumerate(trainLoader):
		
		inputs = inputs.cuda() if useGpu else inputs
		targets = targets.cuda() if useGpu else targets
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		trainLoss = trainLoss + loss.item()
		_, pred = outputs.max(1)
		total += targets.size(0)
		correct += pred.eq(targets).sum().item()
		msg = 'Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})'.format(trainLoss / (batchIdx + 1), 100. * correct / total, correct, total)
		progress_bar(batchIdx, len(trainLoader), msg)
		
	return trainLoss / (batchIdx + 1), 100. * correct / total

def test(epoch, useGpu):
	global bestAcc
	net.eval()
	testLoss = 0
	correct = 0
	total = 0
	for batchIdx, (inputs, targets) in enumerate(valLoader):
		inputs = inputs.cuda() if useGpu else inputs
		targets = targets.cuda() if useGpu else targets
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		testLoss += loss.item()
		_, pred = outputs.max(1)
		total += targets.size(0)
		correct += pred.eq(targets).sum().item()
		
		msg = 'Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d})'.format(testLoss / (batchIdx + 1), 100. * correct / total, correct, total)
		progress_bar(batchIdx, len(valLoader), msg)
		
	# Save checkpoint.
	acc = 100.*correct/total
	if acc > bestAcc:
		
		print ('Saving')
		torch.save(net.state_dict(), os.path.join(args.outDir, 'net.pth'))
		bestAcc = acc
	msg = 'Best Performance: {:.3f}'.format(bestAcc)
	print(msg)
	return testLoss/(batchIdx+1), acc

history = {'trainAcc':[], 'valAcc':[], 'trainLoss':[], 'valLoss':[]}

for epoch in range(args.nbEpoch):
	trainLoss, trainAcc = train(epoch, args.cuda)
	with torch.no_grad() : 
		valLoss, valAcc = test(epoch, args.cuda)
	
	history['trainAcc'].append(trainAcc)
	history['trainLoss'].append(trainLoss)
	
	history['valAcc'].append(valAcc)
	history['valLoss'].append(valLoss)
	
	with open(os.path.join(args.outDir, 'history.json'), 'w') as f : 
		ujson.dump(history, f)


