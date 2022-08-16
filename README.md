# PASTEL
Code for "Position-aware Structure Learning for Graph Topology-imbalance by Relieving Under-reaching and Over-squashing" (accepted by CIKM 2022).

## Overview
- main.py: getting started.
- src/model.py: model configurations.
- src/model_handler.py: overall framework of PASTEL. 
- src/models/graph_clf.py: configurations for GNN backbones and the graph structure learners.
- src/layers/backbones.py: the basic GNNs we used as the backbone, including GCN, GAT, APPNP and GraphSAGE.
- src/layers/graph_learner.py: graph learner of the input graph.
- src/utils/*.py: utils for data preprocessing, loading and other functions.

## Requirements
The implementation of PASTEL is tested under Python 3.8.12, with the following packages installed:
* `dgl==0.8.0.post1`
* `dgl_cu111==0.7.2`
* `networkx==2.6.3`
* `numpy==1.20.3`
* `PyYAML==6.0`
* `scikit_learn==1.1.1`
* `scipy==1.7.1`
* `torch==1.9.1+cu111`
* `torch_geometric==2.0.1`

You can install the above packages by pip:
* `pip install -r requirements.txt`

Or by conda:
* `conda install --yes --file requirements.txt`

## Datasets
All the datasets (i.e., Cora, Citeseer, Photo, Actor) are provided by [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets). 

## Run the codes
Train and evaluate the model:
* `python main.py -config config/<dataset>.yml`  

We train the PASTEL with GNN backbones, and report the `training loss`, `validation loss`, `training accuracy`, `validation accuracy` and the `test accuracy`.
For instance:
* `python main.py -config config/cora.yml` 

```
Train Epoch [ 50 / 10000] | Loss: 5.94542 | NLOSS = -5.94542 | ACC = 0.60714
  Val Epoch [ 50 / 10000] | Loss: 4.11519 | NLOSS = -4.11519 | ACC = 0.63333
Train Epoch [100 / 10000] | Loss: 98.20475 | NLOSS = -98.20475 | ACC = 0.80714
  Val Epoch [100 / 10000] | Loss: 36.39100 | NLOSS = -36.39100 | ACC = 0.82381
Train Epoch [150 / 10000] | Loss: 9.87782 | NLOSS = -9.87782 | ACC = 0.89286
  Val Epoch [150 / 10000] | Loss: 5.40413 | NLOSS = -5.40413 | ACC = 0.80952
Train Epoch [200 / 10000] | Loss: 3.95029 | NLOSS = -3.95029 | ACC = 0.83571
  Val Epoch [200 / 10000] | Loss: 2.70414 | NLOSS = -2.70414 | ACC = 0.82857
Train Epoch [250 / 10000] | Loss: 2.22446 | NLOSS = -2.22446 | ACC = 0.85000
  Val Epoch [250 / 10000] | Loss: 1.83093 | NLOSS = -1.83093 | ACC = 0.84286
Train Epoch [300 / 10000] | Loss: 1.50548 | NLOSS = -1.50548 | ACC = 0.89286
  Val Epoch [300 / 10000] | Loss: 1.61285 | NLOSS = -1.61285 | ACC = 0.83333
......
Finished Training. Training time: 1930.99
********************** MODEL SUMMARY **********************
Best epoch = 707 | NLOSS = -1.11036 | ACC = 0.85238
******************** END MODEL SUMMARY ********************
[test] | ACC: 0.82619 | W-F1: 0.82415 | M-F1: 0.81198
Finished Testing. Testing time: 0.12
```
