python train.py --outDir BSyntheticTh1T3 --cuda --tmpTrainDir BSyntheticTh1TMP --K 300 --margin 3 --tolerance 3. --searchDir ../data/watermark/B_cross_domain_plus/train --queryDir ../data/watermark/B_cross_domain_plus/train --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --labelInfo ../data/BTrainSyntheticRot.json

python train.py --outDir BEngravingTh1T3 --cuda --tmpTrainDir BEngravingTh1TMP --K 300 --margin 3 --tolerance 3. --searchDir ../data/watermark/B_cross_domain_plus/train --queryDir ../data/watermark/B_cross_domain_plus/train --valK 100 --nbSearchImgEpoch 1000 --tripleLossThreshold 1.0 --labelInfo ../data/BTrainEngravingRot.json





