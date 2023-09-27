
## Forensic facial reconstruction: reconstructing the facial profile from the underlying skull


### Overview

| **Title:**    | Automatic Forensic Facial Reconstruction  |
| -------- | ------- |
| **Link to paper:** | [Paper](https://arxiv.org/abs/2308.16139)    |
| **Benchmark:**    | Forensic Facial Reconstruction Benchmark    |
| **Link to benchmark dataset:**    | [Download](https://uni-duisburg-essen.sciebo.de/s/Oz8QmrAUNSPpzub/download)    |
| **Data structure:**| voxel occupancy grid  |


#### Data creation: extract the skull and facial structures (skin, fat, etc) from a whole-body segmentation


![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/forensic_facial_reconstruction.png)



#### Example reconstruction on a test skull

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/facial_reconstruction_results.png)


#### What if the skull is damaged? Repair the skull first before facial reconstruction


![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/skull_reconstruction.png)



### Bibtex
If you use the dataset, please cite the following paper:

```
@article{li2020baseline,
  title={MedShapeNet - A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and others},
  journal={arXiv preprint arXiv:2308.16139},
  year={2023}
}
```
