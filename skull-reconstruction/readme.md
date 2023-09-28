## Skull reconstruction: restoring the full skull when facial bones, cranium, etc are damaged  


#### Overview

| **Title:**    | Automatic Skull Reconstruction |
| -------- | ------- |
| **Link to paper:** | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001251)   <br> [Paper](https://link.springer.com/chapter/10.1007/978-3-031-25046-0_7)   |
| **Benchmark:**    | Skull Reconstruction   |
| **Link to benchmark dataset:**    | [Download](https://uni-duisburg-essen.sciebo.de/s/04LtiVuuUxL4ybT) (facial training and test) <br> [Download](https://dl.dropboxusercontent.com/s/2v09h0vt0k3x9l3/training_set.zip?dl=0) (cranial training) <br> [Download](https://dl.dropboxusercontent.com/s/me3yh4azub7jbpn/test_set_for_participants.zip?dl=0) (cranial test 1) <br> [Download](https://dl.dropboxusercontent.com/s/7ijvewjw4lnjyjv/additional_test_set_for_participants.zip?dl=0) (cranial test 2)  <br> [Download](https://figshare.com/articles/dataset/MUG500_Repository/9616319?file=29011878) (cranial test 3  (craniotomy))|
| **Data structure:**| voxel occupancy grid  |

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/skull_reconstruction.png)



#### Data Creation: Remove (part of) facial bones or cranium  from healthy skulls



#### Methods: Use U-Net style networks that take partial skulls as input and the original skulls as the ground truth  






















#### Bibtex
```
@article{li2021autoimplant,
  title={AutoImplant 2020-first MICCAI challenge on automatic cranial implant design},
  author={Li, Jianning and Pimentel, Pedro and Szengel, Angelika and Ehlke, Moritz and Lamecker, Hans and Zachow, Stefan and Estacio, Laura and Doenitz, Christian and Ramm, Heiko and Shi, Haochen and others},
  journal={IEEE transactions on medical imaging},
  volume={40},
  number={9},
  pages={2329--2342},
  year={2021},
  publisher={IEEE}
}

@article{li2023towards,
  title={Towards clinical applicability and computational efficiency in automatic cranial implant design: An overview of the AutoImplant 2021 cranial implant design challenge},
  author={Li, Jianning and Ellis, David G and Kodym, Old{\v{r}}ich and Rauschenbach, Laur{\`e}l and Rie{\ss}, Christoph and Sure, Ulrich and Wrede, Karsten H and Alvarez, Carlos M and Wodzinski, Marek and Daniol, Mateusz and others},
  journal={Medical Image Analysis},
  pages={102865},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{li2022training,
  title={Training $\beta$-VAE by aggregating a learned Gaussian posterior with a decoupled decoder},
  author={Li, Jianning and Fragemann, Jana and Ahmadi, Seyed-Ahmad and Kleesiek, Jens and Egger, Jan},
  booktitle={MICCAI Workshop on Medical Applications with Disentanglements},
  pages={70--92},
  year={2022},
  organization={Springer}
}

@article{li2020baseline,
  title={MedShapeNet - A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and others},
  journal={arXiv preprint arXiv:2308.16139},
  year={2023}
}
```
