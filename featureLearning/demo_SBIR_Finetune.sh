python train.py --outDir chairsFinetuneT4 --cuda --tmpTrainDir chairsFinetuneTMP --K 100 --margin 3 --tolerance 4. --searchDir ../data/sketch2img/chairs/ --queryDir ../data/sketch2img/chairs/ --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --dataset sketch --modelPth ../model/resnet18ImageNet.pth --labelInfo ../data/chairsTrain.json --featScaleBase 24 --stepNbFeat 4



python train.py --outDir shoesFinetuneT4 --cuda --tmpTrainDir shoesFinetuneTMP --K 100 --margin 3 --tolerance 4. --searchDir ../data/sketch2img/shoes/ --queryDir ../data/sketch2img/shoes/ --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --dataset sketch --modelPth ../model/resnet18ImageNet.pth --labelInfo ../data/shoesTrain.json --featScaleBase 24 --stepNbFeat 4






