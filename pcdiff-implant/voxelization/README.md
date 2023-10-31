# Voxelization
To **train your own models** or **reproduce our results**, please follow the next steps.

## Install the Environment
First you have to ensure that you have all dependencies installed. The simplest way to do so is to use a conda environment (You can use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to create and manage them. If you use Anaconda/Miniconda just replace `mamba` with `conda`).

You can create and activate such an environment called `vox` by running the following command:
```sh
mamba env create -f voxelization/vox_env.yaml
mamba activate vox
```
Next you should install [PyTorch3D](https://pytorch3d.org) **(>=0.5)** following the [official instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). The following command should work:
```sh
mamba install pytorch3d -c pytorch3d
```

The last thing to do is to install [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter):
```sh
mamba install pytorch-scatter -c pyg
```

## Data preprocessing
We provide the scripts `preproc_skullbreak.py` and `preproc_skullfix.py` for preprocessing the required training data. Simply run:
```python
python voxelization/utils/preproc_skullbreak.py
python voxelization/utils/preproc_skullfix.py
```
To disable multiprocessing or change the number of threads use the flag `--multiprocessing` and `--threads`.
To split the training data into a smaller training and an evaluation set (e.g. to allow for early stopping), use the following commands:
```python
python voxelization/utils/split_skullbreak.py
python voxelization/utils/split_skullfix.py
```
## Train the Model
For training new models, we provide the script `train.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python voxelization/train.py configs/train_skullbreak.yaml
```
and the SkullFix dataset:
```python
python voxelization/train.py configs/train_skullfix.yaml
```
The hyperparameters of the model can be adjusted in the corresponding config file. The hyperparameters we used to train our models are already set as default.
## Use the Model
For using the trained model, we provide the script `generate.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python voxelization/generate.py voxelization/configs/gen_skullbreak.yaml
```
and the SkullFix dataset:
```python
python voxelization/generate.py voxelization/configs/gen_skullfix.yaml
```
For changing various parameters the two config files `gen_skullbreak.yaml` and `gen_skullfix.yaml` can be adjusted.
