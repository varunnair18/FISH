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
    PreTrainedModel,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import argparse

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

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', metavar='N', type=str, nargs='+')
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--base_model_idx', type=int, default=0)
    parser.add_argument('--same_classifier', action='store_true', default=False, 
        help='whether the model_name_or_path has the same shape of classifier as which of pretrained_weight')
    parser.add_argument('--diff_aggr_method', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--output_dir', type=str, default='default')

    return parser.parse_args()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    args = create_parser()
    
    base_model_path = args.model_name_or_path[args.base_model_idx]

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_path)

    # save the weight difference
    weight_diff = {}

    for path in args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(path)
        diff = {}
        for n, p in model.named_parameters():
            # the classifiers have different shapes
            if "classifier" in n and not args.same_classifier:
                continue
            pretrained_p = pretrained_model.state_dict()[n]
            diff[n] = p - pretrained_p

        if len(weight_diff) == 0:
            weight_diff.update(diff)
        else:
            for n, p in diff.items():
                weight_diff[n] += diff[n]

    if args.diff_aggr_method == 'mean':
        for n, p in weight_diff.items():
            weight_diff[n] = weight_diff[n] / len(args.model_name_or_path)

    # the final model: calculate the final weights and save in the model
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path)

    for n, p in model.named_parameters():
        if n not in weight_diff:
            continue

        pretrained_p = pretrained_model.state_dict()[n]
        diff_p = weight_diff[n]

        p.data.copy_(pretrained_p.data + diff_p.data)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    tokenizer.save_pretrained(args.output_dir)

    if not isinstance(model, PreTrainedModel):
        torch.save(model.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
