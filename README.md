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

### Models

Contains models for self-supervised pre-training (`SiameseNetwork`) and finetuning 
(`ClassificationNetwork`) and their building blocks.

# Network architecture

![image info](./pictures/network.png)

Siamese network consists of two networks with the same architecture. ResNet-18 (<img src="https://render.githubusercontent.com/render/math?math=W^{x}_{enc}">) as encoder, which is supposed to create hidden features and a projector head <img src="https://render.githubusercontent.com/render/math?math=W^{x}_{pro}">, which is a two layer MLP, with purpose to map the feature space into a lower dimensional hidden space. The online network also has an additional predictor head, again consisting of a two layer MLP. The target network has a <i>StopGrad</i> function instead of a predictor head. Therefore during back propagation, only the weights of the online network are updated. The loss between the output of the online and target network is equal to the cosine-similarity loss function. Note, that the final loss of one image is the symmetric loss <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(\hat{Z}^{(O)}_1, \hat{Z}^{(T)}_2) + \mathcal{L}(\hat{Z}^{(O)}_2, \hat{Z}^{(T)}_1)">, since each augmentation is given to both networks.

# Experiments

### SimSiam with symmetric predictor

Stable version of SimSiam with symmetric predictor (with different learning rate and weight decay for predictor and the rest of the network) can be found on branch 
`simsiam_predictor`.

