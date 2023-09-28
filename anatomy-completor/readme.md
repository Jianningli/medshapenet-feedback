
### Multi-class Anatomy Completor: Generating whole-body pseudo labels given partial/sparse manual annotations

#### Overview

| **Title:**    | A Multi-class Completion Framework for 3D Anatomy Reconstruction |
| -------- | ------- |
| **Link to paper:** | [Paper](https://arxiv.org/abs/2309.04956)    |
| **Benchmark:**    | Multi-class Anatomy Completor    |
| **Link to benchmark dataset:**    |   [Download](https://uni-duisburg-essen.sciebo.de/s/YTotoXPOwzANnXl)  |
| **Data structure:**| voxel occupancy grid  |


#### Dataset creation: 

* Single-class dataset: treat the 104 anatomies as a whole (class '1'), and remove anatomies based on volume ratios

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/single_class_dataset.png)

* Multi-class dataset: extract 12 anatomies and randomly remove some of them to create partial labels

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/completor_dataset.png)

#### Methods:  A 3D U-Net style network, which takes partial and full labels as input and ground truth, respectively
Use the network to learn:<br>
* a many-to-one mapping: multiple sets of partial labels correspond to one set full labels
* a one-to-one residual mapping: one set of partial labels correspond to one set of missing labels

#### Example results 


* Single-class reconstruction:
![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/single_class_results.png)


* Multi-class reconstruction:

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/completor_results.png)



#### Bibtex
If you use the codes and/or dataset, please cite the following papers:

```
@article{li2023anatomy,
      title={Anatomy Completor: A Multi-class Completion Framework for 3D Anatomy Reconstruction}, 
      author={Jianning Li and Antonio Pepe and Gijs Luijten and Christina Schwarz-Gsaxner and Jens Kleesiek and Jan Egger},
      journal={arXiv preprint arXiv:2309.04956},
      year={2023}
}

@article{li2023medshapenet,
  title={MedShapeNet--A Large-Scale Dataset of 3D Medical Shapes for Computer Vision},
  author={Li, Jianning and Pepe, Antonio and Gsaxner, Christina and Luijten, Gijs and Jin, Yuan and Ambigapathy, Narmada and Nasca, Enrico and Solak, Naida and Melito, Gian Marco and Memon, Afaque R and others},
  journal={arXiv preprint arXiv:2308.16139},
  year={2023}
}
```

