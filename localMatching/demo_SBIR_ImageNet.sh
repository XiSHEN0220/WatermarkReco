
python localMatching.py --modelPth ../model/resnet18ImageNet.pth --labelJson ../data/shoesTest.json --searchDir ../data/sketch2img/shoes/ --queryDir ../data/sketch2img/shoes/  --dataset sketch --tolerance 4. --featScaleBase 24  --saveRefFeat --outJson shoesImageNet.json --stepNbFeat 4 

python localMatching.py --modelPth ../model/resnet18ImageNet.pth --labelJson ../data/chairsTest.json --searchDir ../data/sketch2img/chairs/ --queryDir ../data/sketch2img/chairs/  --dataset sketch --tolerance 4. --featScaleBase 24 --saveRefFeat --outJson chairsImageNet.json --stepNbFeat 4

