
## Anatomy Completor


### Overview

**Title:**: A Multi-class Completion Framework for 3D Anatomy Reconstruction  <br> 
**Link to paper:** https://arxiv.org/abs/2309.04956 <br> 
**Shape queries:**<br>
**Link to benchmark dataset:** <br> 
**Data structure:** <ins>voxel occupancy grid</ins> | point cloud | mesh | others <br>


### Code


#### derive the benchmark dataset
**12 anatomies (i.e., 12 classes):** lung, heart, spleen, stomach, pancreas, spine, rib, cage, liver, kidney, aorta, autochthon muscles, pulmonary artery

![Alt text](./assests/multi_class_anatomy.png)

**create the benchmark dataset:** create 10 incomplete sets (input) by removing random anatomies from the 12 anatomies (ground truth)   <br>
**training samples:** 18x10=180 <br> 
**test samples:** 27x10=270

![Alt text](./assests/completor.png)



#### derive the benchmark dataset



#### results

(left: input, right: reconstruction, shown in both 3D and coronal views)

![Alt text](./assests/results.png)





### Bibtex


```
@article{li2023anatomy,
      title={Anatomy Completor: A Multi-class Completion Framework for 3D Anatomy Reconstruction}, 
      author={Jianning Li and Antonio Pepe and Gijs Luijten and Christina Schwarz-Gsaxner and Jens Kleesiek and Jan Egger},
      journal={arXiv preprint arXiv:2309.04956},
      year={2023}
}
```

