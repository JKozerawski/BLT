# BLT: Balancing Long-Tailed Datasets with Adversarially-Perturbed Images
ACCV 2020 [Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Kozerawski_BLT_Balancing_Long-Tailed_Datasets_with_Adversarially-Perturbed_Images_ACCV_2020_paper.pdf), [Supplemental material](https://openaccess.thecvf.com/content/ACCV2020/supplemental/Kozerawski_BLT_Balancing_Long-Tailed_ACCV_2020_supplemental.pdf) and [Video](https://youtu.be/stMSOwrJToU)

Original PyTorch implementation of the ACCV 2020 paper: BLT: Balancing Long-Tailed Datasets with Adversarially-Perturbed Images

## Data

To recreate the experiments from the paper please download the original datasets from:
* [ImageNet](http://image-net.org)
* [Places-365](http://places2.csail.mit.edu)
* [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/2018)

In 'main.py' modify the path to the datasets:
```
data_root = {'ImageNet': '/putYourPathHere/ImageNet_LT',
             'Places': '/putYourPathHere/Places/',
             'iNaturalist_2018': '/putYourPathHere/iNaturalist_2018'}
```

Exact train/val/test splits for all three datasets:
* Can be found [here](https://drive.google.com/file/d/1__ks7IDzoRrRfSPRehX0D2Fk0o4tAg1n/view?usp=sharing)

Please extract them into the 'data' directory:
```
data
  |--ImageNet_LT
    |--ImageNet_LT_train.txt
    |--ImageNet_LT_test.txt
    |--ImageNet_LT_val.txt
  |--Places_LT
    |--Places_LT_train.txt
    |--Places_LT_test.txt
    |--Places_LT_val.txt
  |--iNaturalist_2018
    |--iNaturalist_2018_train.txt
    |--iNaturalist_2018_test.txt
    |--iNaturalist_2018_val.txt
```


## Train

### ImageNet-LT

To train both Stage 1 and Stage 2 run:
```
sh train.sh
```

To train just Stage 1:
```
python3 main.py --config ./config/ImageNet_LT/stage_1.py --no_sampler --no_hallucinations --gpu 0
```

And then to train Stage 2:
```
python3 main.py --config ./config/ImageNet_LT/stage_2.py --gpu 0
```

### Places-LT

To train just Stage 1:
```
python3 main.py --config ./config/Places_LT/stage_1.py --no_sampler --no_hallucinations --gpu 0
```

And then to train Stage 2:
```
python3 main.py --config ./config/Places_LT/stage_2.py --gpu 0
```

### iNaturalist-2018

To train just Stage 1:
```
python3 main.py --config ./config/iNaturalist_2018/stage_1.py --no_sampler --no_hallucinations --gpu 0
```

And then to train Stage 2:
```
python3 main.py --config ./config/iNaturalist_2018/stage_2.py --gpu 0
```

## General information

### Authors:
* Jedrzej Kozerawski (jkozerawski@ucsb.edu)
* Victor Fragoso (victor.fragoso@microsoft.com)
* Nikolaos Karianakis (nikolaos.karianakis@microsoft.com)
* Gaurav Mittal (gaurav.mittal@microsoft.com)
* Matthew Turk (mturk@ttic.edu)
* Mei Chen (mei.chen@microsoft.com)

### Citation

If you use this code for your research, please cite the following paper:
```
[bibtex]
@InProceedings{Kozerawski_2020_ACCV,
    author    = {Kozerawski, Jedrzej and Fragoso, Victor and Karianakis, Nikolaos and Mittal, Gaurav and Turk, Matthew and Chen, Mei},
    title     = {BLT: Balancing Long-Tailed Datasets with Adversarially-Perturbed Images},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```

### Video
YouTube: https://youtu.be/stMSOwrJToU
