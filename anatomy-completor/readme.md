
## Anatomy Completor

### Overview

| **Title:**    | A Multi-class Completion Framework for 3D Anatomy Reconstruction |
| -------- | ------- |
| **Link to paper:** | [Paper](https://arxiv.org/abs/2309.04956)    |
| **Benchmark:**    | Forensic Facial Reconstruction Benchmark    |
| **Link to benchmark dataset:**    |     |
| **Data structure:**| voxel occupancy grid  |

### Code


#### (1) Derive the benchmark dataset
**12 anatomies (i.e., 12 classes):** lung, heart, spleen, stomach, pancreas, spine, rib, cage, liver, kidney, aorta, autochthon muscles, pulmonary artery
Create 10 incomplete sets (input) by removing random anatomies from the 12 anatomies (ground truth). **training samples:** 18x10=180, **test samples:** 27x10=270

![Alt text](./assests/multi_class_anatomy.png)


![Alt text](./assests/completor.png)



#### (2) Train a 3D deep model



#### (3) Results

(left: input, right: reconstruction, shown in both 3D and coronal views)

![Alt text](./assests/results.png)





### Bibtex
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

