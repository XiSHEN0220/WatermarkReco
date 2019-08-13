
python localMatching.py --modelPth ../model/ShoesFinetuned443.pth --labelJson ../data/shoesTest.json --searchDir ../data/sketch2img/shoes/ --queryDir ../data/sketch2img/shoes/  --dataset sketch --tolerance 4. --scaleImgRef 24  --featScaleBase 12 --saveRefFeat --outDir OutShoesFinetune

python localMatching.py --modelPth ../model/ChairsFinetuned918.pth --labelJson ../data/chairsTest.json --searchDir ../data/sketch2img/chairs/ --queryDir ../data/sketch2img/chairs/  --dataset sketch --tolerance 4. --scaleImgRef 24  --featScaleBase 12 --saveRefFeat  --outDir OutChairsFinetune

