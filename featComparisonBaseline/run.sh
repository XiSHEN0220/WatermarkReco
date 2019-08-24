

for scale in 10 12 14 16 18 20 22
do 
python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv5 --imgScale $scale --flip --saveRefFeat --featName AvgPool

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv5 --imgScale $scale  --flip --saveRefFeat --featName Cat

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv5 --imgScale $scale  --flip --saveRefFeat --featName LocalSimi

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv4 --imgScale $scale --flip --saveRefFeat --featName AvgPool

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv4 --imgScale $scale  --flip --saveRefFeat --featName Cat

python featComparisonBaseline.py --modelPth ../model/net98_8.pth --labelJson ../data/BTestSyntheticRot.json --searchDir ../data/watermark/B_cross_domain_plus/val/ --queryDir ../data/watermark/B_cross_domain_plus/val/ --outResJson resBSynthetic.json --featLayer conv4 --imgScale $scale  --flip --saveRefFeat --featName LocalSimi

done



