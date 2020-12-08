# Entity-Graph-VLN

Code of the NeurIPS 2020 paper:
**Language and Visual Entity Relationship Graph for Agent Navigation**<br>
[**Yicong Hong**](http://www.yiconghong.me/), Cristian Rodriguez-Opazo, [Yuankai Qi](https://sites.google.com/site/yuankiqi/home), [Qi Wu](http://www.qi-wu.me/), [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/)<br>

[[Paper](https://papers.nips.cc/paper/2020/hash/56dc0997d871e9177069bb472574eb29-Abstract.html)] [[Supplemental](https://papers.nips.cc/paper/2020/file/56dc0997d871e9177069bb472574eb29-Supplemental.pdf)] [[GitHub](https://github.com/YicongHong/Entity-Graph-VLN)]

<p align="center">
<img src="teaser/f1.png" width="100%">
</p>

## Prerequisites

### Installation

Install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).

Please find the versions of packages in our environment [here](https://github.com/YicongHong/Entity-Graph-VLN/blob/master/entity_graph_vln.yml). In particular, we use:
- Python 3.6.9
- NumPy 1.18.1
- OpenCV 3.4.2
- PyTorch 1.3.0
- Torchvision 0.4.1
- Cuda 10.0

### Data Preparation

Please follow the instructions below to prepare the language and visual data in folders:

- `connectivity`: Download the [connectivity maps](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- `data`:
    - Download the [R2R data](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
    - Download the vocabulary and the augmented data from [EnvDrop](https://github.com/airsplay/R2R-EnvDrop/tree/master/tasks/R2R/data).
<!-- - `img_features`: Download the [Scene features](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365). Download the pre-processed [Object features and vocabulary](). Download -->



Still updating README ...



## Citation
If you use or discuss our Entity Relationship Graph, please cite our paper:
```
@article{hong2020language,
  title={Language and Visual Entity Relationship Graph for Agent Navigation},
  author={Hong, Yicong and Rodriguez, Cristian and Qi, Yuankai and Wu, Qi and Gould, Stephen},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
