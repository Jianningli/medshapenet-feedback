# MedShapeNetCore: [[GitHub](https://github.com/Jianningli/medshapenet-feedback/tree/main), [Release page](https://pypi.org/project/MedShapeNetCore/), [Zenodo](https://zenodo.org/records/10423181), [Publication](https://arxiv.org/abs/2308.16139)]

MedShapeNetCore is a subset of [MedShapeNet](https://arxiv.org/abs/2308.16139), containing more lightweight 3D anatomical shapes in the format of mask, point cloud and mesh. The shape data are stored as numpy arrays in nested dictonaries in *npz* format ([Zenodo](https://zenodo.org/records/10423181)).
This API provides means to downloading, accessing and processing the shape data via Python, which integrates MedShapeNetCore seamless into Python-based machine learning workflows.


# Installation (Python >=3.8, [Release page](https://pypi.org/project/MedShapeNetCore/)) 

    pip install MedShapeNetCore

or install from source:

    python setup.py install
    

# Getting started ([![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jianningli/medshapenet-feedback/blob/main/pip_install_MedShapeNetCore/getting_started.ipynb))

basic commands:

     python -m MedShapeNetCore info  # check the general information of the dataset 
     python -m MedShapeNetCore download DATASET # download a dataset (replace DATASETA with the one you want to download e.g.,  ASOCA)
     python -m MedShapeNetCore check_available_keys DATASET # check the available keys of the DATASET

how to import module functions in python:

     from MedShapeNetCore.MedShapeNetCore import MyDict,MSNLoader,MSNVisualizer,MSNSaver,MSNTransformer
     
For more commands and detailed usage, please refer to the colad [notebook](https://colab.research.google.com/github/Jianningli/medshapenet-feedback/blob/main/pip_install_MedShapeNetCore/getting_started.ipynb).



# Use MedShapeNetCore in Monai

* 3D Shape Classification ([![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jianningli/medshapenet-feedback/blob/main/pip_install_MedShapeNetCore/examples/MONAI_3D_Shape_Classification.ipynb))
