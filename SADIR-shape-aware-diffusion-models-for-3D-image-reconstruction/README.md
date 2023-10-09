# SADIR: Shape-Aware Diffusion Models for 3D Image Reconstruction
This repository provides the PyTorch implementation of the paper: [SADIR: Shape-Aware Diffusion Models for 3D Image Reconstruction](https://arxiv.org/abs/2309.03335)

### Abstract:
3D image reconstruction from a limited number of 2D images has been a long-standing challenge in computer vision and image analysis. While deep learning-based approaches have achieved impressive performance in this area, existing deep networks often fail to effectively utilize the shape structures of objects presented in images. As a result, the topology of reconstructed objects may not be well preserved, leading to the presence of artifacts such as discontinuities, holes, or mismatched connections between different parts. In this paper, we propose a shape-aware network based on diffusion models for 3D image reconstruction, named SADIR, to address these issues. In contrast to previous methods that primarily rely on spatial correlations of image intensities for 3D reconstruction, our model leverages shape priors learned from the training data to guide the reconstruction process. To achieve this, we develop a joint learning network that simultaneously learns a mean shape under deformation models. Each reconstructed image is then considered as a deformed variant of the mean shape. We validate our model, SADIR, on both brain and cardiac magnetic resonance images (MRIs). Experimental results show that our method outperforms the baselines with lower reconstruction error and better preservation of the shape structure of objects within the images.

![SADIR Network](network_architecture.png)

### Citation:
If you use our code, please cite: \
            @misc\{jayakumar2023sadir, \
                  title={SADIR: Shape-Aware Diffusion Models for 3D Image Reconstruction}, \
                  author={Nivetha Jayakumar and Tonmoy Hossain and Miaomiao Zhang},\
                  year={2023},\
                  eprint={2309.03335},\
                  archivePrefix={arXiv},\
                  primaryClass={cs.CV}\
            \}

### Usage
Listed below are the requirements to set up the environment in order to train SADIR.
##### Setup: 
- Lagomorph (https://github.com/jacobhinkle/lagomorph)
- PyTorch 1.12.0
- GCC 11.2.0
- CUDA 11.4.2 

#### Referenced Data Repositories:
[MedShapeNet](https://github.com/Jianningli/medshapenet-feedback) 

### References
[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233): Part of the diffusion model code is from this paper. \
[A Diffusion Model Predicts 3D Shapes from 2D Microscopy Images](https://arxiv.org/abs/2208.14125) \
[Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers](https://proceedings.neurips.cc/paper_files/paper/2022/file/b328c5bd9ff8e3a5e1be74baf4a7a456-Paper-Conference.pdf)
