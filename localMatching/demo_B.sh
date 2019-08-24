python localMatching.py --labelJson ../data/BTestEngravingRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/  --flip --saveRefFeat 

python localMatching.py --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/  --flip --saveRefFeat 

python localMatching.py --labelJson ../data/BTestEngravingRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/  --flip --saveRefFeat --modelPth ../model/EngravingFinetuned75.pth 

python localMatching.py --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/  --flip --saveRefFeat --modelPth ../model/SyntheticFinetuned83.pth
