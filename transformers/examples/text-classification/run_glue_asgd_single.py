# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import torch
from torch.utils.data import DataLoader
from custom_trainer import ASGDTrainer
from copy import deepcopy

os.environ["WANDB_DISABLED"] = "true"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data

    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        logits = return_dicts["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


def create_mask_gradient(model, train_dataset, data_collator, num_samples, keep_ratio, sample_type, grad_type, split=2):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
    )

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    keep_num = split * keep_num

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    # random_indices = torch.randperm(len(top_pos), device=cuda_device)
    
    # Actually is interleave, just not change the naming used before
    random_indices = [torch.arange(i, len(top_pos), split) for i in range(split)]
    random_indices = torch.hstack(random_indices)

    random_indices = random_indices.reshape(split, -1)

    mask_list = []

    for i in range(random_indices.shape[0]):
        # get the indices of the the split
        split_indices = random_indices[i]
        split_top_pos = top_pos[split_indices]
        
        masks = torch.zeros_like(tensors, device=cuda_device)

        masks[split_top_pos] = 1

        assert masks.sum() == len(split_top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0
        
        for k, v in mask_dict.items():
            if "classifier" in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        
        print(pretrain_weight_size, classifier_size, all_params_size)
        print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

        mask_list.append(mask_dict)

    return mask_list


def create_mask_deviation(model, original_model, keep_ratio):
    device = list(model.parameters())[0].device

    deviations = {}
    for name, param in model.named_parameters():
        original_param = original_model[name]
        deviations[name] = torch.abs(param.data - original_param.data.to(device))

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in deviations.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_mask_random(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.rand(param.shape).to(original_device)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=original_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_mask_bias(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    mask_dict = {}

    for name, param in model.named_parameters():
        if "classifier" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        elif "bias" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        else:
            mask_dict[name] = torch.zeros_like(param, device=original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    bias_params_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            bias_params_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(bias_params_size, classifier_size, all_params_size)

    print(f"trainable parameters: {(bias_params_size + classifier_size) / all_params_size * 100} %")

    model.to(original_device)

    return mask_dict


def merge_models(models, merged_model, diff_aggr_method, same_classifier):
    # save the weight difference
    weight_diff = {}

    for model in models:
        diff = {}
        for n, p in model.named_parameters():
            # the classifiers have different shapes
            if "classifier" in n and not same_classifier:
                continue
            pretrained_p = merged_model.state_dict()[n]
            diff[n] = p - pretrained_p

        if len(weight_diff) == 0:
            weight_diff.update(diff)
        else:
            for n, p in diff.items():
                weight_diff[n] += diff[n]

    if diff_aggr_method == 'mean':
        for n, p in weight_diff.items():
            weight_diff[n] = weight_diff[n] / len(models)

    for n, p in merged_model.named_parameters():
        if n not in weight_diff:
            continue

        diff_p = weight_diff[n]

        p.data.copy_(p.data + diff_p.data)


@dataclass
class SparseUpdateTrainingArguments(TrainingArguments):
    num_samples: int = field(
        default=1024,
        metadata={"help": "The number of samples to compute parameters importance"}
    )
    keep_ratio: float = field(
        default=0.005,
        metadata={"help": "The trainable parameters to total parameters."}
    )
    mask_method: str = field(
        default="label-absolute",
        metadata={"help": "The method to select trainable parameters. Format: sample_type-grad_type, \
                   where sample_type in \{label, expect\}, and grad_type in \{absolute, square\}"}
    )
    split: int = field(
        default=2,
        metadata={"help": "The number of mask to generate."}
    )
    merge_step: int = field(
        default=10,
        metadata={"help": "The number of steps to merge the separate models once."}
    )
    same_classifier: bool = field(
        default=False,
        metadata={"help": "whether the model_name_or_path has the same shape of"
                          "classifier as which of pretrained_weight"}
    )
    diff_aggr_method: str = field(
        default='sum'
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_load_path_second: str = field(
        default="",
        metadata={"help": ""}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SparseUpdateTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None or data_args.test_file is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    sample_type, grad_type = training_args.mask_method.split("-")

    masks = create_mask_gradient(
        model,
        train_dataset,
        data_collator,
        training_args.num_samples,
        training_args.keep_ratio,
        sample_type,
        grad_type,
        training_args.split
    )

    # Create Trainer for asynchronous workers
    trainer = ASGDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        merge_step=training_args.merge_step,
        mask=masks[0]
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        for train_results in zip(trainer.train()):
            # re-calulate masks
            masks = create_mask_gradient(
                trainer.model,
                train_dataset,
                data_collator,
                training_args.num_samples,
                training_args.keep_ratio,
                sample_type,
                grad_type,
                training_args.split
            )

            # re-assign the masks and weights
            trainer.set_weight_mask(trainer.model.state_dict(), masks[0])

            tasks = [data_args.task_name]
            train_datasets = [train_dataset]

            # save the train and eval results
            global_step = trainer.state.global_step

            trainer.model.to("cuda")
            for train_datasets, task in zip(train_datasets, tasks):
                train_result = trainer.evaluate(eval_dataset=train_datasets)

                output_train_file = os.path.join(training_args.output_dir, f"train_results_{task}_{global_step}.txt")
                if trainer.is_world_process_zero():
                    with open(output_train_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in sorted(train_result.items()):
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")

            tasks = [data_args.task_name]
            eval_datasets = [eval_dataset]

            for eval_dataset, task in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}_{global_step}.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in sorted(eval_result.items()):
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")

            trainer.model.to("cpu")

        if trainer.is_world_process_zero():
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
