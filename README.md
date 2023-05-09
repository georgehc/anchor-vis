# Code for the paper "A General Framework for Visualizing Embedding Spaces of Neural Survival Analysis Models Based on Angular Information"

Author: George H. Chen (georgechen [at symbol] cmu.edu)

This code accompanies the paper:

> George H. Chen. "A General Framework for Visualizing Embedding Spaces of Neural Survival Analysis Models Based on Angular Information". CHIL 2023.

## Datasets

All datasets we use are standard publicly available datasets and when a dataset involves people, the dataset has already been de-identified (not by us but by whoever provides the data; our secondary analysis of this data is not considered human subject research and does not require IRB approval). In fact, these datasets have all been used in other machine learning papers as well. As far as we are aware, there is no offensive content in the public datasets that we use.

Specifically, our paper uses the following datasets:

- SUPPORT dataset: https://hbiostat.org/data/
- Rotterdam/GBSG data: https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/gbsg
- Survival MNIST: the MNIST dataset itself is just imported using standard PyTorch code and then we provide code that generates the survival labels (following Polsterl 2019 and Goldstein et al 2020; see our paper for details)

## Code dependencies

Our code uses Anaconda Python 3.9 and PyTorch 1.13: please follow the instructions on their official websites for how to install them for your particular machine.

Additional software dependencies: lifelines and pycox. These can both be installed via pip (`pip install lifelines` and `pip install pycox`).

For the optimal regression tree, we acquired an academic license from Interpretable AI (https://www.interpretable.ai/); this only affects the very end of our tabular dataset Jupyter notebooks.

## How to run our code

Our visualizations are all made in Jupyter notebooks. These notebooks can be found in the directory `./notebooks/` (the notebooks should be run from within the `./notebooks/` directory). Specifically, we provide the following notebooks (note that we only provide pre-trained DeepSurv models for the tabular datasets as the model and auxiliary files needed for the Survival MNIST dataset are quite large):

1. [using this notebook does *not* require re-training the DeepSurv model first] `SUPPORT DeepSurv (norm 1 constraint).ipynb`
2. [using this notebook does *not* require re-training the DeepSurv model first] `SUPPORT DeepSurv (norm 1 constraint) - extra - different numbers of clusters.ipynb`  (this notebook is nearly the same as the first one and still uses a mixture of von Mises-Fisher distributions to cluster on embedding vectors, but we explore different choices for the numbers of clusters)
3. [using this notebook does *not* require re-training the DeepSurv model first] `SUPPORT DeepSurv (norm 1 constraint) - extra - different numbers of clusters GMM.ipynb`  (this notebook is like the previous one but instead uses a Gaussian mixture model to cluster on embedding vectors)
4. [using this notebook does *not* require re-training the DeepSurv model first] `SUPPORT DeepSurv (no norm 1 constraint).ipynb`
5. [using this notebook does *not* require re-training the DeepSurv model first] `Rotterdam-GBSG (norm 1 constraint).ipynb`
6. [this dataset requires re-training the DeepSurv model first] `Survival MNIST hypersphere (norm 1 constraint).ipynb`

To train the missing Survival MNIST DeepSurv model from scratch, run the bash script `./train_models/demo_survival_mnist.sh` (this should be run from within the directory `./train_models/`; the hyperparameter grids used are stored in `./train_models/config_image_hypersphere.ini` file). To re-train the models that we have already trained for you, run `./train_model/demo_tabular.sh` (this should also be run from within the directory `./train_models/`; the hyperparameter grids used are stored in `./train_models/config_tabular*.ini` files).

## Compute environment details

All computation except for training models on image datasets was done on a server running Ubuntu 22.04.1 LTS and featuring an Intel Core i9-10900K CPU (10 cores, 20 threads), 64GB RAM, and an Nvidia Quadro RTX 4000 GPU (GPU RAM: 8GB RAM). The Anaconda version used was from October 2022, the scikit-learn version used was 1.0.2, and the PyTorch version was 1.13.0 with CUDA 11.7.
