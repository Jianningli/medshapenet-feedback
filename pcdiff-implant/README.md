# Point Cloud Diffusion Models for Automatic Implant Generation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://pfriedri.github.io/pcdiff-implant-io/)
[![arXiv](https://img.shields.io/badge/arXiv-2303.08061-b31b1b.svg)](https://arxiv.org/abs/2303.08061)

**This is a copy of the [pcdiff-implant](https://github.com/pfriedri/pcdiff-implant) repository** and contains the official PyTorch implementation of the MICCAI 2023 paper [Point Cloud Diffusion Models for Automatic Implant Generation](https://pfriedri.github.io/pcdiff-implant-io/) by Paul Friedrich, Julia Wolleb, Florentin Bieder, Florian M. Thieringer and Philippe C. Cattin.

If you use this code, please consider to :star: **star the [original repository](https://github.com/pfriedri/pcdiff-implant)** and :memo: **cite the paper**:
```bibtex
@InProceedings{10.1007/978-3-031-43996-4_11,
                author="Friedrich, Paul and Wolleb, Julia and Bieder, Florentin and Thieringer, Florian M. and Cattin, Philippe C.",
                title="Point Cloud Diffusion Models for Automatic Implant Generation",
                booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
                year="2023",
                pages="112--122",
               }
```
## Paper Abstract
Advances in 3D printing of biocompatible materials make patient-specific implants increasingly popular. The design of these implants is, however, still a tedious and largely manual process. Existing approaches to automate implant generation are mainly based on 3D U-Net architectures on downsampled or patch-wise data, which can result in a loss of detail or contextual information. Following the recent success of Diffusion Probabilistic Models, we propose a novel approach for implant generation based on a combination of 3D point cloud diffusion models and voxelization networks. Due to the stochastic sampling process in our diffusion model, we can propose an ensemble of different implants per defect from which the physicians can choose the most suitable one. We evaluate our method on the SkullBreak and SkullFix dataset, generating high-quality implants and achieving competitive evaluation scores.

![](./media/overview_pipeline.png)

## Data
We trained our network on the publicly available parts of the [SkullBreak/SkullFix](https://www.sciencedirect.com/science/article/pii/S2352340921001864) datasets.
The method can easily be adapted to work with other datasets (like MedShapeNet).
The used data is available at:
* SkullBreak: https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip
* SkullFix: https://files.icg.tugraz.at/f/2c5f458e781a42c6a916/?dl=1 (we just use the data in ```training_set.zip```)

The provided code works for the following data structure:
```
datasets
└───SkullBreak
    └───complete_skull
    └───defective_skull
        └───bilateral
        └───frontoorbital
        └───parietotemporal
        └───random_1
        └───random_2   
    └───implant
        └───bilateral
        └───frontoorbital
        └───parietotemporal
        └───random_1
        └───random_2
└───SkullFix
    └───complete_skull
    └───defective_skull
    └───implant
```

## Training & Using the Networks
Both networks, the point cloud diffusion model and the voxelization network are trained independently:
* Information on training and using the point cloud diffusion model can be found [here](./pcdiff/README.md)
* Information on training and using the voxelization network can be found [here](./voxelization/README.md)

## Results & Comparing Methods
For further information on the reached evaluation scores and implementation details for our comparing methods, please refer to the **[original repository](https://github.com/pfriedri/pcdiff-implant)**.

## Runtime & GPU memory requirement information
In the following table, we present detailed runtime, as well as GPU memory requirement information. All values have been measured on a system with an AMD EPYC 7742 CPU and an NVIDIA A100 (40GB) GPU.

| Dataset (Method)                     |Point Cloud Diffusion Model | Voxelization Network | Total Time |
|--------------------------------------|----------------------------|----------------------|------------|
| SkullBreak (without ensembling, n=1) |~ 979 s, 4093 MB            |~ 23 s, 12999 MB      |~ 1002 s    |
| SkullBreak (with ensembling, n=5)    |~ 1101 s, 12093 MB          |~ 41 s, 12999 MB      |~ 1142 s    |
| SkullFix (without ensembling, n=1)   |~ 979 s, 4093 MB            |~ 92 s, 12999 MB      |~ 1071 s    |
| SkullFix (with ensembling, n=5)      |~ 1101 s, 12093 MB          |~ 109 s, 12999 MB     |~ 1210 s    |

Generating implants for the SkullFix dataset takes longer, as the volume output by the voxelization network (512 x 512 x 512) needs to be resampled to the initial volume size (512 x 512 x Z), which varies for different implants.

## Acknowledgements
Our code is based on/ inspired by the following repositories:
* https://github.com/autonomousvision/shape_as_points (published under [MIT license](https://github.com/autonomousvision/shape_as_points/blob/main/LICENSE))
* https://github.com/alexzhou907/PVD (published under [MIT license](https://github.com/alexzhou907/PVD/blob/main/LICENSE))

The code for computing the evaluation scores is based on:
* https://github.com/OldaKodym/evaluation_metrics
