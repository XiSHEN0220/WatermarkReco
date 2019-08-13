from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def TrainLoader(batchSize, imgDir, trainTransform) : 
	
	dataloader = DataLoader(ImageFolder(imgDir, trainTransform), batch_size=batchSize, shuffle=True)
	
	return dataloader

def ValLoader(batchSize, imgDir, valTransform) : 
	
	dataloader = DataLoader(ImageFolder(imgDir, valTransform), batch_size=batchSize, shuffle=False)
	
	return dataloader


