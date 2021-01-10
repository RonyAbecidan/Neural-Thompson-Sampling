## Study of the Neural Thompson Sampling algorithm

This repository is the result of an assignment for the "Sequential Decision Making" class of the University of Lille's Msc. in Data Science.

#### Authors :
- Etienne Levecque
- Rony Abecidan

Here, you can find our study about the paper **"Neural Thompson Sampling"** by Weitong Zhang, Dongruo Zhou, Lihong Li and Quanquan Gu published in October 2020. 

In the paper, the authors propose a new strategy enabling to solve the famous contextual bandit problem linking deep learning methods with the Thompson Sampling strategy. They have shown both theoretically and empirically that this strategy can reveal itself to be among the best ones for solving this problem with a competitive regret bound.

***

This repo is made of three parts :

- The article studied in a .pdf format

- A short report discussing about the strategy proposed in the paper with some additional information enable to better understand it.

- An illustrative notebook in which we proposed some experiments enabling to :

    - Understand to what extent the linear strategies are limited
    - Justify the choice of a neural network for approximating the mean rewards
    - See how well the NeuralTS works in practice with simulations and what are its limitations.


## Installation

If you want to test our implementation of NeuralTS you'll have to install the requirements listed in requirements.txt

```bash
pip install -r requirements.txt
```

Please note that a part of the code is largely inspired by a library for MAB problems designed by our teacher [Emilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/).
