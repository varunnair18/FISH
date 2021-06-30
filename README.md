# Fisher Induced Sparse uncHanging (FISH) Mask

This repo contains the code for Fisher Induced Sparse uncHanging (FISH) Mask training, from "Training Neural Networks with Fixed Sparse Masks" by Yi-Lin Sung, Varun Nair, and Colin Raffel. 

**Abstract**: During typical gradient-based training of deep neural networks, all of the model's parameters are updated at each iteration. Recent work has shown that it is possible to update only a small subset of the model's parameters during training, which can alleviate storage and communication requirements. In this paper, we show that it is possible to induce a fixed sparse mask on the model’s parameters that selects a subset to update over many iterations. Our method constructs the mask out of the  parameters with the largest Fisher information as a simple approximation as to which parameters are most important for the task at hand. In experiments on parameter-efficient transfer learning and distributed training, we show that our approach matches or exceeds the performance of other methods for training with sparse updates while being more efficient in terms of memory usage and communication costs.

## Setup

```
pip install supplementary_material/transformers/.
pip install datasets torch==1.8.0 tqdm torchvision==0.9.0
```

## FISH Mask: GLUE Experiments
To run the FISH Mask on a GLUE dataset, code can be run with the following format:

```
$ bash supplementary_material/transformers/examples/text-classification/scripts/run_sparse_updates.sh <dataset-name> <seed> <top_k_percentage> <num_samples_for_fisher>
```

An example command used to generate Table 1 in the paper is as follows, where all GLUE tasks are provided at a seed of 0 and a FISH mask sparsity of 0.5%.

```
bash run_glue_sparse_update.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.005 1024
```


## FISH Mask: CIFAR10 Experiments

[TODO]

## Replicating Results

Replicating each of the tables and figures present in the original paper can be done by running the following: 

```
# Table 1 - Parameter Efficient Fine-Tuning on GLUE

$ bash supplementary_material/transformers/examples/text-classification/scripts/run_table_1.sh
```
```
# Figure 2 - Mask Sparsity Ablation and Sample Ablation

$ bash supplementary_material/transformers/examples/text-classification/scripts/run_figure_2.sh
```
```
# Table 2 - Distributed Training on GLUE

$ bash supplementary_material/transformers/examples/text-classification/scripts/run_table_2.sh
```
```
# Table 3 - Distributed Training on CIFAR10

$ bash supplementary_material/cifar10-fast/scripts/distributed_training.sh

```
```
# Table 4 - Efficient Checkpointing

$ bash supplementary_material/cifar10-fast/scripts/small_checkpoints.sh
```

## Notes
* For reproduction of Diff Pruning results from Table 1, see code [here](https://github.com/varunnair18/DiffPruning/tree/multi-gpus). 


## Acknowledgements

[TODO]

