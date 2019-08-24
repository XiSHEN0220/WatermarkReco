

for t in 3 4 5 22
do
python train.py --outDir BSyntheticTh1Tolerance$t --cuda --tmpTrainDir BSyntheticTMP --K 300 --margin 3 --tolerance $t --searchDir ../data/watermark/B_cross_domain_plus/train --queryDir ../data/watermark/B_cross_domain_plus/train --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --labelInfo ../data/BTrainSyntheticRot.json


python ../localMatching/localMatching.py --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/  --flip --saveRefFeat --modelPth BSyntheticTh1Tolerance$t/net.pth
done





