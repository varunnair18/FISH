import sys
import os

import numpy as np

def extract_loss_n_acc(_file, keyword):
    try:
        with open(_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.rstrip()
                if keyword in line:
                    acc = float(line.split(" ")[-1])
                if "eval_loss" in line:
                    loss = float(line.split(" ")[-1])

        return acc
    
    except:
        return "-"


def cumsom_avg_std(input_list):

    available_list = [acc for acc in input_list if acc != "-"]

    if len(available_list) == 0:
        return "-"

    # print(len(available_list))

    mean = np.mean(available_list)
    std = np.std(available_list)

    return f"{mean:.4f}"


task = sys.argv[1]

if task in ("mnli-mm", "mnli"):
    sub_task = task
    task = "mnli"
else:
    sub_task = task

print(sub_task)

folders = [
    ("densely update single model", [f"tmp/{task}_fully_normal@{seed}" for seed in range(0)]),
    
]

for merge_step in [10, 30, 100]:
    folders.append(
        ("sparsely updated asgd", [f"tmp/{task}_sparse_asgd_merge{merge_step}@0"] + \
        [f"tmp/{task}_sparse_asgd_merge{merge_step}_s2@{seed}" for seed in [1, 2, 4, 5]])
    )
    folders.append(
        ("densely updated asgd (sum)", [f"tmp/{task}_fully_asgd_sum_merge{merge_step}@0"] + \
        [f"tmp/{task}_fully_asgd_sum_merge{merge_step}_s2@{seed}" for seed in [1, 2, 4, 5]])
    )
    folders.append(
        ("sparsely updated asgd (random)", [f"tmp/{task}_random_asgd_sum_merge{merge_step}@0"] + \
        [f"tmp/{task}_random_asgd_sum_merge{merge_step}_s2@{seed}" for seed in [1, 2, 4, 5]])
    )
    

if task in ["rte", "sst2", "qnli", "mnli"]:
    keywords = ["eval_accuracy"]
elif task in ["cola"]:
    keywords = ["eval_matthews_correlation"]
elif task in ["qqp"]:
    keywords = ["eval_f1", "eval_accuracy"]
elif task in ["stsb"]:
    keywords = ["eval_spearmanr", "eval_pearson"]
elif task in ["mrpc"]:
    keywords = ["eval_accuracy", "eval_f1"]


out_format = "{:>30}" * (len(keywords) + 1)

print(out_format.format("-", *keywords))
for name, _folder in folders:

    eval_results_list = [[] for _ in keywords]
    for k, folder in enumerate(_folder):

        try:
            files = os.listdir(folder)

            get_key = lambda name: int(name.split("_")[-1].split(".")[0])
            eval_files = [os.path.join(folder, f) for f in files if f.startswith(f"eval_results_{sub_task}_")]
            eval_files.sort(key=get_key)

            for i, keyword in enumerate(keywords):
                eval_results = [extract_loss_n_acc(f, keyword) for f in eval_files]

                # print(eval_results)
                eval_results_list[i].append(eval_results[-1])
            # print(k, len(eval_files))
        
        except:
            for i, keyword in enumerate(keywords):
                eval_results_list[i].append("-")

    # print(eval_results_list)

    eval_results_list = [cumsom_avg_std(eval_list) for eval_list in eval_results_list]

    print(out_format.format(name, *eval_results_list))