# WatermarkReco

Pytorch implementation of Paper "Large-Scale Historical Watermark Recognition: dataset and a new consistency-based approach"

[[PDF]](http://imagine.enpc.fr/~shenx/Watermark/watermarkReco.pdf) [[Project website]](http://imagine.enpc.fr/~shenx/Watermark) 

<p align="center">
<img src="https://github.com/XiSHEN0220/WatermarkReco/blob/master/figure/teaser.jpg" width="800px" alt="teaser">
</p>

The project is an extension work to [ArtMiner](http://imagine.enpc.fr/~shenx/ArtMiner/). If our project is helpful for your research, please consider citing : 

```
@inproceedings{shen2019watermark,
          title={Large-Scale Historical Watermark Recognition: dataset and a new consistency-based approach},
          author={Shen, Xi and Pastrolin, Ilaria and Bounou, Oumayma and Gidaris, Spyros and Smith, Marc and Poncet, Olivier and Aubry, Mathieu},
          booktitle={Arxiv},
          year={2019}
        }
```

## Table of Content
* [Installation](#installation)
* [Classification](#classification)
* [Local Matching](#local-matching)
* [Feature Learning](#feature-learning)
* [Visual Results](#visual-results)



## Installation

### Dependencies

Code is tested under **Pytorch > 1.0 + Python 3.6** environment. To install all dependencies : 

``` Bash
bash requirement.sh
```

### Datasets

To download datasets: 
``` Bash
cd data/
bash download_data.sh
```

Resume of the watermark dataset:
 
| Dataset |   #cls X #img per cls | Description and Task| Our Top-1 Acc (%)|
| :------: |  :------: | :------: |:------: |
| A Train | 100 X 50 + 100 X 10| Train and test on the same 100 classes, classification|-|
| A Test | 100 X 3| Another 100 classes different from A train, 1 ref + 2 test photographs, one-shot recognition|90|
| B Train | 140 X 1-7| 140 classes, 1 drawing + 1-7 photographs, cross-domain feature fine-tuning|-|
| B Test | 100 X 3| 100 classes different from B Train, 1 drawing + 2 photographs, one-shot cross-domain recognition|83|
| Briquet | 16,753 X 1| 16, 753 classes, 1 engraving, large scale one-shot cross-domain recognition|55|

Resume of the shoes / chairs dataset:
 
| Dataset |  #cls X #img per cls | Description and Task| Our Top-1 Acc (%)|
| :------: |  :------: | :------: |:------: |
| Shoes Train| 304 X 2| 1 photo + 1 sketch, cross-domain feature fine-tuning |-|
| Shoes Test | 115 X 2| Another 115 classes different from Shoes train, 1 photo + 1 sketch, one-shot cross-domain recognition|52.2|
| Chair Train | 200 X 2| 200 classes, 1 photo + 1 sketch, cross-domain feature fine-tuning|-|
| Chair Test | 97 X 2| Another 97 classes different from Chair Train, 1 photo + 1 sketch, one-shot cross-domain recognition|91.8|

###  Models

To download pretrained models: 
``` Bash
cd model/
bash download_model.sh # classification models + fine-tuned models
```
## Classification

Dataset: A Train

``` Bash
cd classification/
bash demo_train.sh # Training with Dropout Ratio 0.7
```

## Local Matching

### One-shot Recognition 

Dataset: A Test
``` Bash
cd localMatching/
bash demo_A.sh 
```

Feature Similarity Baselines: 
``` Bash
cd featComparisonBaseline/
bash bestParam.sh # Run with resolution 256 * 256
bash run.sh # Run with different resolutions
```

### One-shot Cross-domain Recognition


Dataset: B Test 
``` Bash
cd localMatching/
bash demo_B.sh # Using drawing or synthetic as references with / without finetuned model
```

Dataset: Shoes / Chairs
``` Bash
cd localMatching/
bash demo_SBIR.sh # Evaluate on Shoes and Chairs dataset with / without finetuned model
```

### Large Scale One-shot Cross-domain Recognition (16,753-class)

Dataset: B Test + Briquet 
``` Bash
cd localMatching/
bash demo_Briquet_Baseline.sh # AvgPool, Concat and Local Sim. baselines
bash demo_Briquet_Baseline.sh # AvgPool, Concat and Local Sim. baselines

```
## Feature Learning

Dataset: B Train 
``` Bash
cd featureLearning/
bash demo_B_Finetune.sh # Eta = 3 for both drawing and synthetic references
```

Dataset: Shoes / Chairs 
``` Bash
cd featureLearning/
bash demo_SBIR_Finetune.sh # Eta = 4 for chairs and shoes
```

## Visual Results 

More visual results can be found in our [Project website](http://imagine.enpc.fr/~shenx/Watermark/). 

Top-5 retrieval results on Briquet + B Test dataset with using engraving as references:

<p align="center">
<img src="https://github.com/XiSHEN0220/WatermarkReco/blob/master/figure/engraving.jpg" width="800" alt="teaser">
</p>


Top-5 retrieval results on Briquet + B Test dataset with using synthetic image as references:

<p align="center">
<img src="https://github.com/XiSHEN0220/WatermarkReco/blob/master/figure/synthetic.jpg" width="800px" alt="teaser">
</p>

Top-5 retrieval results on Shoes / Chairs Test dataset:

<p align="center">
<img src="https://github.com/XiSHEN0220/WatermarkReco/blob/master/figure/chairs.jpg" width="800px" alt="teaser">
</p>





























