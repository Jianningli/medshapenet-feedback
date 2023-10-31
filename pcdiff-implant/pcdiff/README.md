# Point Cloud Diffusion Model for Anatomy Restoration
To **train your own models** or **reproduce our results**, please follow the next steps.

## Install the Environment
First you have to ensure that you have all dependencies installed. The simplest way to do so is to use a conda environment (You can use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to create and manage them. If you use Anaconda/Miniconda just replace `mamba` with `conda`).

You can create and activate such an environment called `pcd` by running the following command:
```sh
mamba env create -f pcdiff/pcd_env.yaml
mamba activate pcd
```
## Data preprocessing
Make sure you downloaded the data sets and stored them in the correct folder (see data structure).
We provide the scripts `preproc_skullbreak.py` and `preproc_skullfix.py` to preprocess the datasets.
Simply run the following commands:

```python
python pcdiff/utils/preproc_skullbreak.py
python pcdiff/utils/preproc_skullfix.py
```
To disable multiprocessing or change the number of threads use the flags `--multiprocessing` and `--threads`. 

Preprocessing the data may take some hours (depending on your hardware). We highly recommend to use multiprocessing.

## Train/Test Split
To randomly split the data into a training and a test set run:
```python
python pcdiff/utils/split_skullbreak.py
python pcdiff/utils/split_skullfix.py
```
The script creates a `train.csv` and `test.csv` file in the corresponding folder of the dataset, which can be used for the `--path` flag during training.
## Train the Model
For training new models, we provide the script `train_completion.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python pcdiff/train_completion.py --path datasets/SkullBreak/train.csv --dataset SkullBreak
```
and the SkullFix data set:
```python
python pcdiff/train_completion.py --path datasets/SkullFix/train.csv --dataset SkullFix
```
We provide a lot of different flags to change the hyperparameters of the model (details in the code). The hyperparameters used to generate the presented results are set as default.

## Use the Model
For using a trained model, we provide the script `test_completion.py` and two exemplary commands on how to use it for the SkullBreak:
```python
python pcdiff/test_completion.py --path datasets/SkullBreak/test.csv --dataset SkullBreak --model MODELPATH --eval_path datasets/SkullBreak/results
```
and the SkullFix data set (if you want to use the proposed ensembling method, use the `--num_ens` flag to specifiy the number of different implants to be generated):
```python
python pcdiff/test_completion.py --path datasets/SkullFix/test.csv --dataset SkullFix --num_ens 5 --model MODELPATH --eval_path datasets/SkullFix/results
