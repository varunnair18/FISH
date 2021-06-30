# Fisher Induced Sparse uncHanging (FISH) Mask

This repo contains the code for Fisher Induced Sparse uncHanging (FISH) Mask training, from "Training Neural Networks with Fixed Sparse Masks" by Yi-Lin Sung, Varun Nair, and Colin Raffel. 

**Abstract**: During typical gradient-based training of deep neural networks, all of the model's parameters are updated at each iteration. Recent work has shown that it is possible to update only a small subset of the model's parameters during training, which can alleviate storage and communication requirements. In this paper, we show that it is possible to induce a fixed sparse mask on the model’s parameters that selects a subset to update over many iterations. Our method constructs the mask out of the  parameters with the largest Fisher information as a simple approximation as to which parameters are most important for the task at hand. In experiments on parameter-efficient transfer learning and distributed training, we show that our approach matches or exceeds the performance of other methods for training with sparse updates while being more efficient in terms of memory usage and communication costs.

## Setup

```
pip install transformers/.
pip install datasets torch==1.8.0 tqdm torchvision==0.9.0
```

## FISH Mask: GLUE Experiments

### Parameter-Efficient Transfer Learning

To run the FISH Mask on a GLUE dataset, code can be run with the following format:

```
$ bash transformers/examples/text-classification/scripts/run_sparse_updates.sh <dataset-name> <seed> <top_k_percentage> <num_samples_for_fisher>
```

An example command used to generate Table 1 in the paper is as follows, where all GLUE tasks are provided at a seed of 0 and a FISH mask sparsity of 0.5%.

```
$ bash transformers/examples/text-classification/scripts/run_sparse_updates.sh "qqp mnli rte cola stsb sst2 mrpc qnli" 0 0.005 1024
```

### Distributed Training 

To use the FISH mask on the GLUE tasks in a distributed setting, one can use the following command.
```
$ bash transformers/examples/text-classification/scripts/distributed_training.sh <dataset-name> <seed> <num_workers> <training_epochs> <gpu_id>
```

Note the `<dataset-name>` here can only contain one task, so an example command could be

```
$ bash transformers/examples/text-classification/scripts/distributed_training.sh "mnli" 0 2 3.5 0
```

## FISH Mask: CIFAR10 Experiments

To run the FISH mask on CIFAR10, code can be run with the following format:

### Distributed Training
```
$ bash cifar10-fast/scripts/distributed_training_fish.sh <num_samples_for_fisher> <top_k_percentage> <training_epochs> <worker_updates> <learning_rate> <num_workers>
```

For example, in the paper, we compute the FISH mask of the 0.5% sparsity level by 256 samples and distribute the job to 2 workers for a total of 50 epochs training. Then the command would be

```
$ bash cifar10-fast/scripts/distributed_training_fish.sh 256 0.005 50 2 0.4 2
```

### Efficient Checkpointing

```
$ bash cifar10-fast/scripts/small_checkpoints_fish.sh <num_samples_for_fisher> <top_k_percentage> <training_epochs> <learning_rate> <fix_mask>
```

The hyperparameters are almost the same as distributed training. However, the `<fix_mask>` is to indicate to fix the mask or not, and a valid input is either 0 or 1 (1 means to fix the mask). 


## Replicating Results

Replicating each of the tables and figures present in the original paper can be done by running the following: 

```
# Table 1 - Parameter Efficient Fine-Tuning on GLUE

$ bash transformers/examples/text-classification/scripts/run_table_1.sh
```
```
# Figure 2 - Mask Sparsity Ablation and Sample Ablation

$ bash transformers/examples/text-classification/scripts/run_figure_2.sh
```
```
# Table 2 - Distributed Training on GLUE

$ bash transformers/examples/text-classification/scripts/run_table_2.sh
```
```
# Table 3 - Distributed Training on CIFAR10

$ bash cifar10-fast/scripts/distributed_training.sh

```
```
# Table 4 - Efficient Checkpointing

$ bash cifar10-fast/scripts/small_checkpoints.sh
```

## Notes
* For reproduction of Diff Pruning results from Table 1, see code [here](https://github.com/varunnair18/DiffPruning/tree/multi-gpus). 


## Acknowledgements

[TODO]

