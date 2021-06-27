import sys
import os

def extract_loss_n_acc(_file, keyword):
    with open(_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            if keyword in line:
                acc = float(line.split(" ")[-1])
            if "eval_loss" in line:
                loss = float(line.split(" ")[-1])

    return acc, loss

task = sys.argv[1]

if task in ("mnli-mm", "mnli"):
    sub_task = task
    task = "mnli"
else:
    sub_task = task

print(sub_task)

folders = [
    ("densely update single model", f"tmp/{task}_fully_normal@0"),
    
]

for merge_step in [10, 30, 100]:
    folders.append(
        ("sparsely updated asgd", f"tmp/{task}_sparse_asgd_merge{merge_step}_s4@0")
    )
    folders.append(
        ("densely updated asgd (sum)", f"tmp/{task}_fully_asgd_sum_merge{merge_step}_s4@0")
    )
    folders.append(
        ("sparsely updated asgd (random)", f"tmp/{task}_random_asgd_sum_merge{merge_step}_s4@0")
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
for name, folder in folders:

    files = os.listdir(folder)

    get_key = lambda name: int(name.split("_")[-1].split(".")[0])
    eval_files = [os.path.join(folder, f) for f in files if f.startswith(f"eval_results_{sub_task}_")]
    eval_files.sort(key=get_key)
    print(len(eval_files))

    eval_results_list = []

    for keyword in keywords:
        eval_results = [extract_loss_n_acc(f, keyword) for f in  eval_files]
        eval_results_list.append(eval_results[-1][0])

    print(out_format.format(name, *eval_results_list))