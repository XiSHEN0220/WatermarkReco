python train.py --outDir chairsFinetuneTolerance4 --cuda --tmpTrainDir chairsFinetuneToleranceTOTO --K 100 --margin 3 --tolerance 4. --searchDir ../data/sketch2img/chairs/ --queryDir ../data/sketch2img/chairs/ --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --dataset sketch --modelPth ../model/resnet18ImageNet.pth --labelInfo ../data/chairsTrain.json --featScaleBase 12 --queryScale 24

python train.py --outDir shoesFinetuneTolerance3 --cuda --tmpTrainDir shoesFinetuneToleranceTOTO --K 100 --margin 3 --tolerance 3. --searchDir ../data/sketch2img/shoes/ --queryDir ../data/sketch2img/shoes/ --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --dataset sketch --modelPth ../model/resnet18ImageNet.pth --labelInfo ../data/shoesTrain.json --featScaleBase 12 --queryScale 24






