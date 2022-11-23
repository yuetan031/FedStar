# FedStar
Implementation of the paper accepted by AAAI 2023: Federated Learning on Non-IID Graphs via Structural Knowledge Sharing.

## Data Preparation
* Download the [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/) manually and put them under ```./Data/``` directory.
* Experiments are run on 16 public graph classification datasets from four different domains, including Small Molecules (MUTAG, BZR, COX2, DHFR, PTC\_MR, AIDS, NCI1), Bioinformatics (ENZYMES, DD, PROTEINS), Social Networks (COLLAB, IMDB-BINARY, IMDB-MULTI), and Computer Vision (Letter-low, Letter-high, Letter-med).

## Running examples
* First, to train on the cross-dataset setting based on seven small molecule datasets (CHEM):
```
python exps/main_multiDS.py --repeat 1 --data_group 'chem' --seed 1 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 2 --data_group 'chem' --seed 2 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 3 --data_group 'chem' --seed 3 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 4 --data_group 'chem' --seed 4 --alg fedstar --type_init 'rw_dg'
python exps/main_multiDS.py --repeat 5 --data_group 'chem' --seed 5 --alg fedstar --type_init 'rw_dg'
```

* After running the above command lines, the raw results are stored in ```./outputs/raw/```.

* Then, to process the raw results:
```
python exps/aggregateResults.py --data_group 'chem'
```

* Finally, the results are stored in ```./outputs/processed/```.

## Options
The default values for various paramters parsed to the experiment are given in ```./exps/main_multiDS.py```. Details about some of those parameters are given here.
* ```--data_group:```  The group of datasets that corresponds to different settings. Default: 'chem'. Options: 'chem', 'biochem', 'biochemsn', 'biosncv'.
* ```--num_rounds:``` The number of rounds for simulation. Default: 200.
* ```--local_epoch:``` The number of local epochs. Default: 1.
* ```--n_rw:``` The dimension of random walk-based structure embedding (RWSE). Default: 16.
* ```--n_dg:``` The dimension of degree-based structure embedding (DSE). Default: 16.
* ```--type_init:``` The type of structure embedding. Default: 'rw_dg'. Options: 'rw', 'dg', 'rw_dg'.
* ```--hidden:``` The number of hidden units. Default: 64.
* ```--nlayer:``` The number of GNN layers. Default: 3.

## Citation
If you find this project helpful, please consider to cite the following paper:
```
@inproceedings{
tan2023federated,
title={Federated Learning on Non-IID Graphs via Structural Knowledge Sharing},
author={Tan, Yue and Liu, Yixin and Long, Guodong and Jiang, Jing and Lu, Qinghua and Zhang, Chengqi},
booktitle={AAAI},
year={2023}
}
```
