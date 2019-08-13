
import os
import torch
from tqdm import tqdm
import torch.utils.data as data
import PIL.Image as Image


def LoadImg(path) :
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, imgDir, nbImg, transform):
        self.imgDir = imgDir
        self.transform= transform
        self.posDir = os.path.join(imgDir, 'posPair')
        self.negDir = os.path.join(imgDir, 'negPair')
        self.nbImg = nbImg
        

    def __getitem__(self, index):

        posI1 = LoadImg(os.path.join(self.posDir, '{}_1.jpg'.format(index)))
        posI2 = LoadImg(os.path.join(self.posDir, '{}_2.jpg'.format(index)))
        negI1 = LoadImg(os.path.join(self.negDir, '{}_1.jpg'.format(index)))
        negI2 = LoadImg(os.path.join(self.negDir, '{}_2.jpg'.format(index)))
        
        return {'posI1' : self.transform(posI1), 'posI2' :self.transform(posI2), 'negI1' :self.transform(negI1), 'negI2' :self.transform(negI2)}

    def __len__(self):
        return self.nbImg

## Train Data loader
def TrainDataLoader(imgDir, nbImg, transform, batchSize):

    trainSet = ImageFolder(imgDir, nbImg, transform)
    trainLoader = data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=1, drop_last = True)

    return trainLoader

