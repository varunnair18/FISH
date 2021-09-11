import numpy as np

def get_acc(file_name):
    try:
        with open(file_name, "r") as f:
            line = f.readlines()[-1]

            acc = float(line.split("\t")[-1])
    except:
        acc = "-"

    return acc


def cumsom_avg_std(input_list):

    available_list = [acc for acc in input_list if acc != "-"]

    if len(available_list) == 0:
        return "-"

    mean = np.mean(available_list)
    std = np.std(available_list)

    return f"{mean:.2f}({std:.2f})"

files = []

xs = ["0.8", "0.4", "0.2", "0.08", "0.04", "0.02"]
ys = ["0.005", "0.02", "0.1", "0.25", "1.0"]

row_format = "{:>15}" * (len(xs) + 1)


epoch = 100

from collections import defaultdict

avg_dict = defaultdict(lambda: defaultdict(list))
for seed in [0, 1, 2, 3, 4, "avg"]:
    print(f"seed = {seed}")
    
    print(row_format.format("-", *xs))

    if seed == "avg":
        for sparsity in ys:
            acc_list = []
            for lr in xs:
                acc_list.append(cumsom_avg_std(avg_dict[sparsity][lr]))

            print(row_format.format(sparsity, *acc_list))

    else:
        for sparsity in ys:
            acc_list = []
            for lr in xs:
                # if lr == "0.4":
                #     file_name = f"logs/su_s1024_k{sparsity}_e{epoch}.tsv"
                # else:
                if sparsity == "1.0":
                    file_name = f"logs/su_resnet34_k{sparsity}_e{epoch}_l{lr}@{seed}.tsv"
                else:
                    file_name = f"logs/su_resnet34_s1024_k{sparsity}_e{epoch}_l{lr}_w0_fix@{seed}.tsv"

                acc = get_acc(file_name)

                acc_list.append(acc)

                avg_dict[sparsity][lr].append(acc)


            print(row_format.format(sparsity, *acc_list))
