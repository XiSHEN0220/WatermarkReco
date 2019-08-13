python feat_cos_avg_match_briquet.py --modelPth ../model/SyntheticFinetuned77.pth --labelJson ../data/BTestBriquet.json --searchDir ../data/watermark/briquet_synthetic/ --outResJson BriquetLocalSimiRetrievalRes.json --nbSaveRefFeat 1000 

python localMatching_briquet.py --modelPth ../model/SyntheticFinetuned77.pth --labelJson ../data/BTestBriquet.json --preOrderJson BriquetLocalSimiRetrievalRes.json --evaluateTopK 1000 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetSyntheticFinetune.json


