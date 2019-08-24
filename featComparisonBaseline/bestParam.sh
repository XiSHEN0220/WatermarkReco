

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/labelAOneshot.json --searchDir ../data/watermark/A_one_shot_recognition/ --queryDir ../data/watermark/A_one_shot_recognition/ --outResJson resA.json --featLayer conv5 --imgScale 14 --flip --saveRefFeat --featName AvgPool

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/labelAOneshot.json --searchDir ../data/watermark/A_one_shot_recognition/ --queryDir ../data/watermark/A_one_shot_recognition/ --outResJson resA.json --featLayer conv5 --imgScale 14 --flip --saveRefFeat --featName Cat

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/labelAOneshot.json --searchDir ../data/watermark/A_one_shot_recognition/ --queryDir ../data/watermark/A_one_shot_recognition/ --outResJson resA.json --featLayer conv5 --imgScale 14 --flip --saveRefFeat --featName LocalSimi


python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestEngravingRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBEngraving.json --featLayer conv5 --imgScale 14 --flip --saveRefFeat --featName AvgPool

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestEngravingRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBEngraving.json --featLayer conv4 --imgScale 14  --flip --saveRefFeat --featName Cat

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestEngravingRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBEngraving.json --featLayer conv4 --imgScale 14  --flip --saveRefFeat --featName LocalSimi


python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv5 --imgScale 14 --flip --saveRefFeat --featName AvgPool

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv4 --imgScale 14  --flip --saveRefFeat --featName Cat

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv4 --imgScale 14  --flip --saveRefFeat --featName LocalSimi




