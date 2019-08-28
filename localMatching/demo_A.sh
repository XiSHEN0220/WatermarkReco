python localMatching.py --featScaleBase 16 --stepNbFeat 2 --nbscale 2 --margin 2 --tolerance 1.5 --labelJson ../data/labelAOneshot.json --searchDir ../data/watermark/A_one_shot_recognition/ --queryDir ../data/watermark/A_one_shot_recognition/ --outJson ourAOneshot.json --flip --saveRefFeat --featLayer conv4

python localMatching.py --labelJson ../data/labelAOneshot.json --searchDir ../data/watermark/A_one_shot_recognition/ --queryDir ../data/watermark/A_one_shot_recognition/ --outDir visAOneshot --flip --saveRefFeat
