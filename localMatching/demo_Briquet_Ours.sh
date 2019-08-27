#python LocalSimiBriquet.py --outResJson LocalSimiBriquet.json --nbSaveRefFeat 1000

#python LocalSimiBriquet.py --outResJson LocalSimiBriquetFinetune.json --nbSaveRefFeat 1000 --modelPth ../model/SyntheticFinetuned83.pth


python localMatching_briquet.py --modelPth ../model/SyntheticFinetuned83.pth --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquetFinetune.json --evaluateTopK 100 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank100SyntheticFinetune.json



python localMatching_briquet.py --modelPth ../model/SyntheticFinetuned83.pth --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquetFinetune.json --evaluateTopK 1000 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank1000SyntheticFinetune.json

python localMatching_briquet.py --modelPth ../model/SyntheticFinetuned83.pth --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquetFinetune.json --evaluateTopK 16000 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank16000SyntheticFinetune.json




python localMatching_briquet.py  --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquet.json --evaluateTopK 100 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank100Synthetic.json


python localMatching_briquet.py  --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquet.json --evaluateTopK 1000 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank1000Synthetic.json

python localMatching_briquet.py  --labelJson ../data/BTestBriquet.json --preOrderJson LocalSimiBriquet.json --evaluateTopK 16000 --searchDir ../data/watermark/briquet_synthetic/ --outJson BriquetRerank16000Synthetic.json



