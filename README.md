# Training Code for IPN

This repository implemented training code for ["Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks"](https://arxiv.org/abs/1904.09791) (CVPR 2019) based on its official demo [code](https://github.com/seoungwugoh/ivs-demo).

## Create the environment

```bash
# create conda env
conda create -n ipn python=3.7
# activate conda env
conda activate ipn
# install pytorch
conda install pytorch=1.3 torchvision
# install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation

- DAVIS 2017 Dataset
  - Download the data and human annotated scribbles [here](https://davischallenge.org/davis2017/code.html).
  - Place `DAVIS` folder into `root/data`.

## Training

```bash
python ipn_trainval.py
```
trained models will be stored in `root/results`

## Evaluation
The evaluation code can be found [here](https://github.com/svip-lab/IVOS-W).

## Reference
```bash
Fast User-Guided Video Object Segmentation by Interaction-and-Propagation Networks
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
CVPR 2019
```