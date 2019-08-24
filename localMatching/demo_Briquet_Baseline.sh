python AvgPoolBriquet.py --outResJson AvgPoolBriquet.json --nbSaveRefFeat 1000

python ConcatBriquet.py --outResJson ConcatBriquet.json --nbSaveRefFeat 1000

python LocalSimiBriquet.py --outResJson LocalSimiBriquet.json --nbSaveRefFeat 1000



python LocalSimiBriquet.py --outResJson LocalSimiBriquetFinetune.json --nbSaveRefFeat 1000 --modelPth ../model/SyntheticFinetuned83.pth


python AvgPoolBriquet.py --outResJson AvgPoolBriquetFinetune.json --nbSaveRefFeat 1000 --modelPth ../model/SyntheticFinetuned83.pth


python ConcatBriquet.py --outResJson ConcatBriquetFinetune.json --nbSaveRefFeat 1000 --modelPth ../model/SyntheticFinetuned83.pth
