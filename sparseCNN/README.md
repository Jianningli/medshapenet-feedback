
## [Sparse convolutional neural network for high-resolution skull shape completion and shape super-resolution](https://www.nature.com/articles/s41598-023-47437-6)

Check out the [**Motivation Letter**](https://dl.dropboxusercontent.com/s/2cit5cue7e1u557/notes.txt?dl=0).

Check out the demonstration [**video**](https://www.techrxiv.org/articles/preprint/Sparse_Convolutional_Neural_Networks_for_Medical_Image_Analysis/19137518?file=34041689).

Full texts can be found [**HERE**](https://www.nature.com/articles/s41598-023-47437-6).

**Our paper describes a practical solution to the curse of dimensionality in medical image analysis. The proposed approach is particularly relevant if your GPU memory does not possess the capacity to process the medical images at their original resolution and/or the sluggish training prohibits efficient hyper-parameter tunning. Our work describes the utility of sparse convolutions in shape completion, super-resolution and segmentation tasks. Experiments show that the proposed method is able to process high-resolution medical images using moderate memory and at a high speed.**


## skull shape completion and super-resolution
Thanks to [sparse convolutions](https://nvidia.github.io/MinkowskiEngine/overview.html), a deep neural net can be trained on full-resolution skull images (512x512xZ) for shape completion and shape super-resolution tasks. [Previous approaches](https://ieeexplore.ieee.org/document/9420655) (or [this](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001251)) use dense convolutions, so that the images have to be downsampled to fit in the GPU memory. A super-resolution network upsamples a coarse image to higher resolution (e.g., 512x512xZ) and restores its fine geometric details on the shape surface.

| shape completion (input-prediction-gt)|super-resolution (64-128-256-512)|
| ------      | ------ |
|[![skull shape completion](https://github.com/Jianningli/SparseCNN/blob/main/images/github1.png)] |  [![skull shape super-resolution](https://github.com/Jianningli/SparseCNN/blob/main/images/github2.png)]|



## medical image segmentation

Detailed workflow of using sparse neural nets in medical image **segmentation** can be found [here](https://www.techrxiv.org/articles/preprint/Sparse_Convolutional_Neural_Networks_for_Medical_Image_Analysis/19137518) (**Appendices C**).


| segmentation 1|segmentation 2|
| ------      | ------ |
|[![segmentation](https://github.com/Jianningli/SparseCNN/blob/main/images/github4.png)] |  [![segmentation](https://github.com/Jianningli/SparseCNN/blob/main/images/github5.png)]|




To cite our work:

```
@article{li2023sparse,
title={Sparse Convolutional Neural Network for High-resolution Skull Shape Completion and Shape Super-resolution},
author={Li, Jianning and Gsaxner, Christina and Pepe, Antonio and Schmalstieg, Dieter and Kleesiek, Jens and Egger, Jan},
journal={Scientific Reports},
volume={13},
doi={https://doi.org/10.1038/s41598-023-47437-6},
year={2023}}
```


