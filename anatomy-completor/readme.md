
### Multi-class Anatomy Completor: Generating whole-body pseudo labels given partial/sparse manual annotations (Semantic Scene Completion on Voxel Grids)

#### Overview

| **Title:**    | A Multi-class Completion Framework for 3D Anatomy Reconstruction |
| -------- | ------- |
| **Link to paper:** | [Paper](https://arxiv.org/abs/2309.04956)    |
| **Benchmark:**    | Multi-class Anatomy Completor    |
| **Link to benchmark dataset:**    |   [Download](https://files.icg.tugraz.at/f/b0623306eb9246be8c3c/?dl=1) (multi-class) |
| **Data structure:**| voxel occupancy grid  |
| **Other info:**| [Presentation](https://jianningli.me/pdfs/Anatomy%20completor.pdf) |




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


* Multi-class reconstruction: it also belongs to a semantic scene completion (SSC) problem on other point clouds or voxel grids (related papers [1](https://ojs.aaai.org/index.php/AAAI/article/view/16451),[2](https://openaccess.thecvf.com/content/CVPR2023/papers/Xia_SCPNet_Semantic_Scene_Completion_on_Point_Cloud_CVPR_2023_paper.pdf),[3](https://arxiv.org/abs/2210.05891),[4](https://arxiv.org/pdf/1611.08974.pdf))

![datacreation](https://github.com/Jianningli/medshapenet-feedback/blob/main/assets/completor_results.png)



#### Bibtex
If you find our papers useful to your research and/or use the codes and/or dataset, please cite the following papers:

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

