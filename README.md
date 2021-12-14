<h1 align="center">
  <b>Self-Supervised Learning without contrastive pairs</b><br>
</h1>

<p align="center">
  <i>Tobias Höppe, Agnieszka Miszkurka, Dennis Wilkman</i><br>
</p>

This repo reproduces the results of [Understanding Self-Supervised Learning Dynamics without Contrastive Pairs](https://arxiv.org/pdf/2102.06810.pdf) paper. It is a final project for Advanced Deep Learning course at KTH Royal Institute of Technology in Stockholm.

## Project overview

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

## Experiments

Stable varsion of SimSiam with symmetric predicitor (with different learning rate and weight decay for predictor and the rest of the network) can be found on branch 
`simsiam_predictor`.

