# WatermarkReco

Pytorch implementation of Paper "Large-Scale Historical Watermark Recognition: dataset and a new consistency-based approach"

[[Arxiv]](http://arxiv.org/pdf/1908.10254.pdf) [[Project website]](http://imagine.enpc.fr/~shenx/Watermark) 

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

We release our watermark dataset composed of 4 subsets targeting on 4 different tasks: classification, one-shot, one-shot cross-domain and large-scale one-shot cross-domain recognition. 

You can run the following command to directly download the dataset:
``` Bash
cd data/
bash download_data.sh ## Watermark + Shoes / Chairs datasets
```

Or click [here(~400M)](http://imagine.enpc.fr/~shenx/data/Watermark.zip) to download it. 

A full description of dataset is provided in our [project website](http://imagine.enpc.fr/~shenx/Watermark).


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
demo_Briquet_Ours.sh # Our approach w / wo F.T.

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

More visual results can be found in our [project website](http://imagine.enpc.fr/~shenx/Watermark/). 

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

## Acknowledgment

This work was partly supported by ANR project EnHeritANR-17-CE23-0008 PSL [Filigrane pour tous](https://filigranes.hypotheses.org/english) project and gifts from Adobe to Ecole des Ponts. 



























