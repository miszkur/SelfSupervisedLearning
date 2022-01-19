<h1 align="center">
  <b>Self-Supervised Learning without contrastive pairs</b><br>
</h1>

<p align="center">
  <i>Tobias Höppe, Agnieszka Miszkurka, Dennis Wilkman</i><br>
</p>

This repo reproduces the results of [Understanding Self-Supervised Learning Dynamics without Contrastive Pairs](https://arxiv.org/pdf/2102.06810.pdf) paper. It is a final project for Advanced Deep Learning course at KTH Royal Institute of Technology in Stockholm.

We implemented <b>all-in-one siamese netwok</b> which can work as:

* [BYOL](https://arxiv.org/pdf/2006.07733v3.pdf)
* [SimSiam](https://arxiv.org/pdf/2011.10566.pdf)
* [DirectPred](https://arxiv.org/pdf/2102.06810.pdf)
* [DirectCopy](https://arxiv.org/pdf/2110.04947.pdf)


# Environment

The project is implemented with Tensorflow 2. Prepare an virtual environment with python>=3.6, and then use the following command line for the dependencies.

```
pip install -r requirements.txt
```

# Project overview

The project structure is as follows:

```bash
.
├── data_processing
├── experiments
│   ├── notebooks
│   │   ├── results_eigenspace
│   │   └── saved_model
│   ├── scripts
│   │   ├── results_eigenspace
│   │   └── saved_model
│   └── visualisation
└── models

```

### Data processing
 
Contains augmentations and methods for processing CIFAR-10 and STL-10.

### Experiments

Contains notebooks and scripts for running experiments along with visualisation utilities.

All parameter settigns can be found in `config.py`.

### Models

Contains models for self-supervised pre-training (`SiameseNetwork`) and finetuning 
(`ClassificationNetwork`) and their building blocks.

# Network architecture

![image info](./pictures/network.png)

Siamese network consists of two networks with the same architecture. ResNet-18 (<img src="https://render.githubusercontent.com/render/math?math=W^{x}_{enc}">) as encoder, which is supposed to create hidden features and a projector head <img src="https://render.githubusercontent.com/render/math?math=W^{x}_{pro}">, which is a two layer MLP, with purpose to map the feature space into a lower dimensional hidden space. The online network also has an additional predictor head, again consisting of a two layer MLP. The target network has a <i>StopGrad</i> function instead of a predictor head. Therefore during back propagation, only the weights of the online network are updated. The loss between the output of the online and target network is equal to the cosine-similarity loss function. Note, that the final loss of one image is the symmetric loss <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(\hat{Z}^{(O)}_1, \hat{Z}^{(T)}_2) "> + <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(\hat{Z}^{(O)}_2, \hat{Z}^{(T)}_1) ">, since each augmentation is given to both networks.

# Experiments

## Configuration 

Below are all available configurations which can be found in `config.py`.

|  Network \ Settings | original                             | Symmetry regularisation                 | One layer predictor  (original: two layers) |
|---------------------|--------------------------------------|-----------------------------------------|---------------------------------------------|
| BYOL                | get_byol / get_eigenspace_experiment | get_eigenspace_experiment_with_symmetry | get_byol_baseline                           |
| SimSiam             | get_simsiam                          | get_simsiam_symmetric                   | get_simsiam_baseline                        |


|  Network \ Settings | original        | SimSiam          | 3 layer predictor     |
|---------------------|-----------------|------------------|-----------------------|
| DirectPred          | get_direct_pred | get_simsiam_pred | get_deeper_projection |
| DirectCopy          | get_direct_copy |                  |                       |

### SimSiam with symmetric predictor

Stable (not collapsing) version of SimSiam with symmetric predictor (with different learning rate and weight decay for predictor and the rest of the network) can be found on branch 
`simsiam_predictor`.

### How to run 

For pretraining you can use one of the scripts in `experiments/scripts` e.g. by running

```bash
python -m direct_pred [--epochs num_epochs]
```

Alternatively, you can use jupyter notebook, for example see `experiments/notebooks/direct_pred.ipynb`.
Pretrained model will be saved in `saved_model` directory. There are models already available in those folders.
For supervised fine tuning you can use `classification.ipynb`. 


# Results

For detailed results see report of our project.
All our experiments were run on CIFAR-10 due to computational constraints. 
Self-Supervised pretraining takes around 4 hours 30 minutes on GCP's V100.

<div align="center">

| Model | Config | Accuracy  |
|-------|---------|------------|
| BYOL | get_byol | 85.7% |
| SimSiam | get_simsiam | 79.4%|  


![image info](./pictures/results.png)
Results for DirectPred and DirectCopy with and without EMA. SGD baseline is BYOL with one layer predictor. 

</div>